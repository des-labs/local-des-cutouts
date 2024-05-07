#!/usr/bin/env python3

"""
author: Landon Gelman, 2018-2020
author: T. Andrew Manning, 2020
author: Francisco Paz-Chinchon, 2019
description: command line tools for making large numbers and multiple kinds of cutouts from the Dark Energy Survey catalogs
"""

import os, sys
import argparse
import logging
import glob
import time

# from numpy.core import numeric
import easyaccess as ea
import numpy as np
import pandas as pd
import PIL
import uuid
import json
import yaml
import shlex
import subprocess
from astropy import units
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs import utils
from astropy.visualization import make_lupton_rgb as mlrgb
from mpi4py import MPI as mpi
from PIL import Image
import math
from io import StringIO

STATUS_OK = 'ok'
STATUS_ERROR = 'error'

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = 144000000        # allows Pillow to not freak out at a large filesize
ARCMIN_TO_DEG = 0.0166667        # deg per arcmin
COORD_PRECISION = 1e-6
# TODO: Move the database and release names to environment variables or to a config file instead of hard-coding
VALID_DATA_SOURCES = {
    'DESDR': [
        'DR1',
        'DR2',
    ],
    'DESSCI': [
        'Y6A2',
        'Y3A2',
        # 'Y1A1',
        # 'SVA1',
    ]
}

comm = mpi.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()

class MPILogHandler(logging.FileHandler):
    def __init__(self, filename, comm, amode=mpi.MODE_WRONLY|mpi.MODE_CREATE|mpi.MODE_APPEND):
        self.comm = comm
        self.filename = filename
        self.amode = amode
        self.encoding = 'utf-8'
        logging.StreamHandler.__init__(self, self._open())
    def _open(self):
        stream = mpi.File.Open(self.comm, self.filename, self.amode)
        stream.Set_atomicity(True)
        return stream
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.Write_shared((msg+self.terminator).encode(self.encoding))
        except Exception:
            self.handleError(record)
    def close(self):
        if self.stream:
            self.stream.Sync()
            self.stream.Close()
            self.stream = None

def getPathSize(path):
    dirsize = 0
    for entry in os.scandir(path):
        if entry.is_dir(follow_symlinks=False):
            dirsize += getPathSize(entry.path)
        else:
            try:
                dirsize += os.path.getsize(entry)
            except FileNotFoundError:
                continue

    return dirsize

def _DecConverter(ra, dec):
    ra1 = np.abs(ra/15)
    raHH = int(ra1)
    raMM = int((ra1 - raHH) * 60)
    raSS = (((ra1 - raHH) * 60) - raMM) * 60
    raSS = np.round(raSS, decimals=4)
    raOUT = '{0:02d}{1:02d}{2:07.4f}'.format(raHH, raMM, raSS) if ra > 0 else '-{0:02d}{1:02d}{2:07.4f}'.format(raHH, raMM, raSS)

    dec1 = np.abs(dec)
    decDD = int(dec1)
    decMM = int((dec1 - decDD) * 60)
    decSS = (((dec1 - decDD) * 60) - decMM) * 60
    decSS = np.round(decSS, decimals=4)
    decOUT = '-{0:02d}{1:02d}{2:07.4f}'.format(decDD, decMM, decSS) if dec < 0 else '+{0:02d}{1:02d}{2:07.4f}'.format(decDD, decMM, decSS)

    return raOUT + decOUT

def make_rgb(cutout, rgb_type, color_set, outdir, basename):
    output_files = []
    if len(color_set) != 3:
        logger.error('Exactly three colors are required for RGB generation.')
        return output_files
    # Support color set specification as list of letters or a string
    if isinstance(color_set, list):
        color_set = ''.join(color_set)
    # Ensure that FITS source file has been generated for each required color. This
    # should be unnecessary unless this function is used independently
    fits_filepaths = {}
    for color in color_set:
        # TODO: Consolidate these naming conventions in a dedicated function
        fits_filepath = os.path.join(outdir, basename + '_{}.fits'.format(color))
        fits_filepaths[color] = fits_filepath
        if not os.path.exists(outdir) or not glob.glob(fits_filepath):
            logger.info('The FITS file required for the RGB image was not found as expected. Generating it now...')
            files = make_fits_cut(cutout, color, outdir, basename)
            output_files.extend(files)
            # If the file remains absent, log the error
            if not os.path.exists(outdir) or not glob.glob(fits_filepath):
                logger.error('Error creating the required FITS file "{}".'.format(fits_filepath))
                return output_files
    
    # Output RGB file basepath
    filename_base = '{0}_{1}'.format(basename, color_set)
    fits_file_list = [fits_filepaths[color] for color in fits_filepaths]

    if rgb_type == 'MAKE_RGB_LUPTON':
        # Create output subdirectory
        os.makedirs(outdir, exist_ok=True)
        # TODO: Verify that the comparison of generated size to the requested size is redundant since 
        # this is logged in the FITS cutout file generation
        try:
            r_data = fits.getdata(fits_file_list[0], 'SCI')
            g_data = fits.getdata(fits_file_list[1], 'SCI')
            b_data = fits.getdata(fits_file_list[2], 'SCI')
        except:
            r_data = fits.getdata(fits_file_list[0], 'IMAGE')
            g_data = fits.getdata(fits_file_list[1], 'IMAGE')
            b_data = fits.getdata(fits_file_list[2], 'IMAGE')
        # Generate RGB image from three FITS file data
        image = mlrgb(
            # FITS file data
            r_data, g_data, b_data, 
            # Lupton parameters
            minimum=cutout['RGB_MINIMUM'], 
            stretch=cutout['RGB_STRETCH'], 
            Q=cutout['RGB_ASINH']
        )
        image = Image.fromarray(image, mode='RGB')
        image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        luptonnm = filename_base + '_lupton'
        filename = luptonnm+'.png'
        filepath = os.path.join(outdir, filename)
        image.save(filepath, format='PNG')
        output_files.append(filename)

    elif rgb_type == 'MAKE_RGB_STIFF':
        # Create output subdirectory
        os.makedirs(outdir, exist_ok=True)
        stiffnm = filename_base + '_stiff'
        tiff_filepath = os.path.join(outdir, stiffnm+'.tiff')
        # Call STIFF using the 3 bands.
        cmd_stiff = 'stiff {}'.format(' '.join(fits_file_list))
        cmd_stiff += ' -OUTFILE_NAME {}'.format(tiff_filepath)
        cmd_stiff = shlex.split(cmd_stiff)
        try:
            subprocess.call(cmd_stiff)
        except OSError as e:
            logger.error(e)

        # Convert the STIFF output from tiff to png and remove the tiff file.
        filename = stiffnm+'.png'
        filepath = os.path.join(outdir, filename)
        cmd_convert = 'convert {0} {1}'.format(tiff_filepath, filepath)
        cmd_convert = shlex.split(cmd_convert)
        try:
            subprocess.call(cmd_convert)
            output_files.append(filename)
        except OSError as e:
            logger.error(e)
        try:
            os.remove(tiff_filepath)
        except OSError as e:
            logger.error(e)

    return output_files

def make_fits_cut(cutout, colors, outdir, basename):
    # Array of generated files
    output_files = []
    # logger.info('makefits_cut for cutout: {}'.format(cutout))

    # Iterate over individual colors (i.e. bands)
    for color in colors.lower():
        # Construct the output filename, with different naming scheme based on coord or coadd position type
        filename = basename + '_{}.fits'.format(color)
        filepath = os.path.join(outdir, filename)
        # logger.info('target FITS file path: {}'.format(filepath))
        # If file exists, continue to the next color
        if glob.glob(filepath):
            continue

        # Start a timer for the FITS file open
        start = time.time()
        try:
            # Y-band color must be uppercase; others lowercase
            source_file_color = color.upper() if color.lower() == 'y' else color.lower()
            data_file_search = cutout['TILEDIR'] + '*_{}.fits.fz'.format(source_file_color)
            fits_file = glob.glob(data_file_search)
            # logger.info('data_file_search: {}, fits_file: {}'.format(data_file_search, fits_file))

            hdu_list = fits.open(fits_file[0])
        except IndexError as e:
            print('No FITS file in {0} color band found. Will not create cutouts in this band.'.format(color))
            logger.error('MakeFitsCut - No FITS file in {0} color band found. Will not create cutouts in this band.'.format(color))
            logger.error(f'''{e}''')
            continue        # Just go on to the next color in the list
        
        # # Mark time for FITS file opened
        # end1 = time.time()
        # Iterate over all HDUs in the tile
        new_hdu_list = fits.HDUList()
        pixelscale = None
        for hdu in hdu_list:
            if hdu.name == 'PRIMARY':
                continue
            data = hdu.data
            header = hdu.header.copy()
            wcs = WCS(header)
            cutout_2D = Cutout2D(data, cutout['POSITION'], cutout['SIZE'], wcs=wcs, mode='trim')
            crpix1, crpix2 = cutout_2D.position_cutout
            x, y = cutout_2D.position_original
            crval1, crval2 = wcs.wcs_pix2world(x, y, 1)
            header['CRPIX1'] = crpix1
            header['CRPIX2'] = crpix2
            header['CRVAL1'] = float(crval1)
            header['CRVAL2'] = float(crval2)
            header['HIERARCH RA_CUTOUT'] = cutout['RA']
            header['HIERARCH DEC_CUTOUT'] = cutout['DEC']
            if not new_hdu_list:
                new_hdu = fits.PrimaryHDU(data=cutout_2D.data, header=header)
                pixelscale = utils.proj_plane_pixel_scales(wcs)
            else:
                try:
                    new_hdu = fits.ImageHDU(data=cutout_2D.data, header=header, name=header['EXTNAME'])
                except:
                    new_hdu = fits.ImageHDU(data=cutout_2D.data, header=header, name=header['XTENSION'])
                    
            new_hdu_list.append(new_hdu)
        # Check the size of the cutout compared to the requested size and warn if different
        if pixelscale is not None:
            dx = round(float(cutout['SIZE'][1] * ARCMIN_TO_DEG / pixelscale[0] / units.arcmin))        # pixelscale is in degrees (CUNIT)
            dy = round(float(cutout['SIZE'][0] * ARCMIN_TO_DEG / pixelscale[1] / units.arcmin))
            if (new_hdu_list[0].header['NAXIS1'], new_hdu_list[0].header['NAXIS2']) != (dx, dy):
                logger.info('MakeFitsCut - {} is smaller than user requested. This is likely because the object/coordinate was in close proximity to the edge of a tile. (dx, dy): {}, (NAXIS1, NAXIS2): {}'.format(('/').join(filepath.split('/')[-2:]), (dx, dy), (new_hdu_list[0].header['NAXIS1'], new_hdu_list[0].header['NAXIS2'])))
        # Create output subdirectory
        os.makedirs(outdir, exist_ok=True)
        # Save the resulting cutout to disk
        new_hdu_list.writeto(filepath, output_verify='exception', overwrite=True, checksum=False)
        new_hdu_list.close()
        # Add the output file to the return list
        output_files.append(filename)
        # # Mark time for FITS output file written
        # end2 = time.time()
        # logger.info('Time to open FITS file {} sec. Time to write output FITS file {} sec.'.format(round(float(end1-start), 3), round(float(end2-end1), 3)))

    return output_files

def run(conf):
    # Configure logging
    formatter = logging.Formatter('%(asctime)s - '+str(rank)+' - %(levelname)-8s - %(message)s')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    # Local directory where generated output files will be stored
    if not conf['outdir']:
        logger.info('outdir must be a path.')
        sys.exit(1)
    outdir = os.path.join(conf['outdir'], '')
    # Create the output directory.
    if rank == 0:
        try:
            os.makedirs(outdir, exist_ok=True)
        except OSError as e:
            logger.error(e)
            logger.error('Error creating output directory. Aborting job.')
            sys.stdout.flush()
            comm.Abort()
    # Ensure that all processes wait until the output directory exists
    comm.Barrier()

    # Validate the configuration
    valid, msg = validate_config(conf)
    if not valid:
        logger.error('Invalid config: {}'.format(msg))
        sys.exit(1)

    # Load default values
    with open(os.path.join(os.path.dirname(__file__), 'config.default.yaml'), 'r') as configfile:
        defaults = yaml.load(configfile, Loader=yaml.FullLoader)
    # Locally mounted directory containing tile data files
    if not 'tiledir' in conf:
        conf['tiledir'] = defaults['tiledir']
    elif conf['tiledir'] != 'auto' and not os.path.exists(conf['tiledir']):
        logger.info('tiledir path not found.')
        sys.exit(1)
    # Set a random Job ID if not provided
    if 'jobid' not in conf:
        conf['jobid'] = str(uuid.uuid4())

    # Configure logging to file
    logname = os.path.join(conf['outdir'], 'cutout_{}.log'.format(conf['jobid']))
    fh = MPILogHandler(logname, comm)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Construct the complete cutout request table
    user_df = construct_cutouts_table(conf)

    # Initialize the database connection and basic job info
    # Get job ID from config
    jobid = conf['jobid']
    # Get database connection and cursor objects using easyaccess
    uu = conf['username']
    pp = conf['password']
    if conf['db'].lower() == 'desdr':
        # Use the Oracle service account to access the relevant tile path info table if provided in the config
        if conf['oracle_service_account_db'] and conf['oracle_service_account_user'] and conf['oracle_service_account_pass']:
            db = conf['oracle_service_account_db']
            uu = conf['oracle_service_account_user']
            pp = conf['oracle_service_account_pass']
        else:
            db = conf['db'].lower()
    elif conf['db'].lower() == 'dessci':
        db = conf['db'].lower()
    else:
        logger.error('Invalid database.')
        return
    
    conn = ea.connect(db, user=uu, passwd=pp)
    curs = conn.cursor()

    complete_df = None
    df = None
    split_df = None
    summary = {}
    if rank == 0:
        # Record the configuration of the cutout requests. Be careful to 
        # omit sensitive information like passwords.
        recorded_options = [
            'jobid',
            'username',
            'tiledir',
            'db',
            'release',
            'xsize',
            'ysize',
            'colors_fits',
            'make_fits',
            'make_rgb_lupton',
            'make_rgb_stiff',
            'colors_fits',
            'rgb_stiff_colors',
            'rgb_lupton_colors',
            'rgb_minimum', 
            'rgb_stretch', 
            'rgb_asinh',
            'discard_fits_files',
        ]
        summary = {
            'options': {key: value for (key, value) in conf.items() if key in recorded_options },
            'cutouts': user_df,
        }
        # Start a timer for the database query
        start = time.time()

        # Subset of DataFrame where position type is RA/DEC coordinates
        coord_df = user_df[user_df['POSITION_TYPE'] == 'coord']
        coord_df_columns = ['RA', 'DEC', 'RA_ADJUSTED', 'XSIZE', 'YSIZE']
        if len(coord_df) > 0 and all(k in coord_df for k in coord_df_columns):
            coord_query_df = coord_df[coord_df_columns]
        else:
            coord_query_df = []

        # Subset of DataFrame where position type is Coadd ID
        coadd_df = user_df[user_df['POSITION_TYPE'] == 'coadd']
        coadd_df_columns = ['COADD_OBJECT_ID', 'XSIZE', 'YSIZE']
        if len(coadd_df) > 0 and all(k in coadd_df for k in coadd_df_columns):
            coadd_query_df = coadd_df[coadd_df_columns]
        else:
            coadd_query_df = []

        # Define the temporary database tablename and output CSV filepath
        tablename = 'positions_'+jobid.replace("-","_")
        tablename_csv_filepath = os.path.join(outdir, tablename+'.csv')

        # Define the catalogs to query based on the chosen database
        if conf['db'].upper() == 'DESSCI':
            catalog_coord = 'Y3A2_COADDTILE_GEOM'
            if conf['release'].upper() in ['Y1A1', 'SVA1']:
                catalog_coadd = '{}_COADD_OBJECTS'.format(conf['release'].upper())
                catalog_coadd_id_column_name = 'COADD_OBJECTS_ID'
            elif conf['release'].upper() in ['Y3A2', 'Y6A2']:
                catalog_coadd = '{}_COADD_OBJECT_SUMMARY'.format(conf['release'].upper())
                catalog_coadd_id_column_name = 'COADD_OBJECT_ID'
            else:
                logger.error('Invalid release.')
                sys.exit(1)
        elif conf['db'].upper() == 'DESDR':
            catalog_coord = '{}_TILE_INFO'.format(conf['release'].upper())
            catalog_coadd = '{}_MAIN'.format(conf['release'].upper())
            catalog_coadd_id_column_name = 'COADD_OBJECT_ID'
        else:
            logger.error('Invalid database.')
            sys.exit(1)

        #############################################################
        # Find tile names associated with each position by COORDINATE
        #
        unmatched_coords = {'RA':[], 'DEC':[]}
        
        if len(coord_query_df) > 0:
            # Create the temporary database table from a CSV dump of the DataFrame
            coord_query_df.to_csv(tablename_csv_filepath, index=False)
            conn.load_table(tablename_csv_filepath, name=tablename.upper())

            query = '''
                select temp.RA, temp.DEC, temp.RA_ADJUSTED, temp.RA as ALPHAWIN_J2000, temp.DEC as DELTAWIN_J2000, m.TILENAME, temp.XSIZE, temp.YSIZE
                from {tablename} temp 
                left outer join {catalog} m on 
                (
                    m.CROSSRA0='N' and 
                    (temp.RA between m.URAMIN and m.URAMAX) and 
                    (temp.DEC between m.UDECMIN and m.UDECMAX)
                ) or 
                (
                    m.CROSSRA0='Y' and 
                    (temp.RA_ADJUSTED between m.URAMIN-360 and m.URAMAX) and 
                    (temp.DEC between m.UDECMIN and m.UDECMAX)
                ) and m.ID < 200000
            '''.format(tablename=tablename.upper(), catalog=catalog_coord)
            
            # Overwrite DataFrame with extended table that has tilenames
            coord_query_df = conn.query_to_pandas(query)
            # Drop the temporary table
            curs.execute('drop table {}'.format(tablename.upper()))
            # os.remove(tablename_csv_filepath)

            # Record unmatched positions
            dftemp = coord_query_df[ (coord_query_df['TILENAME'].isnull()) ]
            unmatched_coords['RA'] = dftemp['RA'].tolist()
            unmatched_coords['DEC'] = dftemp['DEC'].tolist()

        #############################################################
        # Find tile names associated with each position by COADD ID
        #
        unmatched_coadds = []

        if len(coadd_query_df) > 0:
            # Create the temporary database table from a CSV dump of the DataFrame
            coadd_query_df.to_csv(tablename_csv_filepath, index=False)
            conn.load_table(tablename_csv_filepath, name=tablename.upper())

            query = '''
                select temp.COADD_OBJECT_ID, temp.XSIZE, temp.YSIZE, m.ALPHAWIN_J2000, m.DELTAWIN_J2000, m.RA, m.DEC, m.TILENAME
                from {tablename} temp 
                left outer join {catalog} m on temp.COADD_OBJECT_ID=m.{catalog_coadd_id_column_name}
            '''.format(tablename=tablename.upper(), catalog=catalog_coadd, catalog_coadd_id_column_name=catalog_coadd_id_column_name)
            
            # Overwrite DataFrame with extended table that has tilenames
            coadd_query_df = conn.query_to_pandas(query)
            coadd_query_df['COADD_OBJECT_ID'] = coadd_query_df['COADD_OBJECT_ID'].astype('str')
            # Drop the temporary table
            curs.execute('drop table {}'.format(tablename.upper()))
            # os.remove(tablename_csv_filepath)

            # Record unmatched positions
            dftemp = coadd_query_df[ (coadd_query_df['TILENAME'].isnull()) | (coadd_query_df['ALPHAWIN_J2000'].isnull()) | (coadd_query_df['DELTAWIN_J2000'].isnull()) | (coadd_query_df['RA'].isnull()) | (coadd_query_df['DEC'].isnull()) ]
            unmatched_coadds = dftemp['COADD_OBJECT_ID'].tolist()

        try:
            # Merge results with the original sub-df
            for row_index, cutout in coord_df.iterrows():
                for param in ['TILENAME', 'ALPHAWIN_J2000', 'DELTAWIN_J2000']:
                    value = coord_query_df.loc[(abs(coord_query_df['RA'] - cutout['RA']) < COORD_PRECISION) & (abs(coord_query_df['DEC'] - cutout['DEC']) < COORD_PRECISION), param]
                    value = value.reset_index(drop=True)
                    if len(value) > 0:
                        coord_df.at[row_index, param] = value[0]
            for row_index, cutout in coadd_df.iterrows():
                for param in ['TILENAME', 'ALPHAWIN_J2000', 'DELTAWIN_J2000', 'RA', 'DEC']:
                    value = coadd_query_df.loc[coadd_query_df['COADD_OBJECT_ID'] == cutout['COADD_OBJECT_ID'], param]
                    value = value.reset_index(drop=True)
                    if len(value) > 0:
                        coadd_df.at[row_index, param] = value[0]
        except Exception as e:
            logger.error(str(e).strip())
            sys.exit(1)

        #############################################################
        # Recombine the subcomponent DataFrames
        #
        complete_df = pd.concat([coord_df, coadd_df])

        # Refine the values
        complete_df = complete_df.replace('-9999',np.nan)
        complete_df = complete_df.replace(-9999.000000,np.nan)
        
        # Drop unmatched entries from the DataFrame
        complete_df = complete_df.dropna(axis=0, how='any', subset=['TILENAME', 'ALPHAWIN_J2000', 'DELTAWIN_J2000'])
        complete_df = complete_df.sort_values(by=['TILENAME'])
        # Requesting multiple cutouts of the same position with different sizes is not allowed.
        complete_df = complete_df.drop_duplicates(['RA','DEC'], keep='first')
        complete_df.reset_index(drop=True)
        logger.info('Complete positions table (rank={}):\n{}'.format(rank, complete_df))
        # Save the cutout positions table to disk
        complete_df.to_csv(tablename_csv_filepath, index=False)

        end1 = time.time()
        query_elapsed = '{0:.2f}'.format(end1-start)
        logger.info('Querying took (s): ' + query_elapsed)
        summary['query_time'] = query_elapsed
        split_df = np.array_split(complete_df, nprocs)

    # Split the table into roughly equal parts and distribute to the parallel processes
    df = comm.scatter(split_df, root=0)
    df.reset_index(drop=True)
    logger.info('Table subset (rank={}):\n {}'.format(rank, df))
    
    all_generated_files = []

    qtemplate = "select FITS_IMAGES from {} where tilename = '{}' and band = 'i'"
    table_path = "MCARRAS2.{}_TILE_PATH_INFO".format(conf['release'])
    # Determine the file paths for each unique relevant tile  
    for tilename in df['TILENAME'].unique():
        try:
            if conf['tiledir'].lower() != 'auto':
                tiledir = os.path.join(conf.tiledir, tilename)
            else:
                dftile = conn.query_to_pandas(qtemplate.format(table_path, tilename))
                tiledir = os.path.dirname(dftile.FITS_IMAGES.iloc[0])
                logger.info(f'tiledir: "{tiledir}"')
                if conf['release'].upper() in ('Y6A2', 'Y3A2', 'DR1', 'DR2'):
                    tiledir = tiledir.replace('https://desar2.cosmology.illinois.edu/DESFiles/desarchive/OPS/', '/des003/desarchive/') + '/'
                elif conf['release'].upper() in ('SVA1', 'Y1A1'):
                    tiledir = tiledir.replace('https://desar2.cosmology.illinois.edu/DESFiles/desardata/OPS/coadd/', '/des004/coadd/') + '/'
                # if conf['release'].upper() in ('Y6A2'):
                #     tiledir = f'''/tiles/dr2/{tilename}/'''
            # Clean up path formatting
            tiledir = os.path.join(tiledir, '')
            # Store tiledir in table 
            df.loc[df['TILENAME'] == tilename, 'TILEDIR'] = tiledir
        except Exception as e:
            logger.error(str(e).strip())

    #############################################################
    # Main iteration loop over all cutout requests
    #
    # Iterate over each row and validate parameter values
    for row_index, cutout in df.iterrows():
        # Collect all generated files associated with this position
        generated_files = []
        discard_filepaths = []
        cutout['SIZE'] = units.Quantity((cutout['YSIZE'], cutout['XSIZE']), units.arcmin)
        cutout['POSITION'] = SkyCoord(cutout['ALPHAWIN_J2000'], cutout['DELTAWIN_J2000'], frame='icrs', unit='deg', equinox='J2000', representation_type='spherical')

        # Files are stored in subdirectories based on the unique position of the cutout request
        if cutout['POSITION_TYPE'] == 'coadd':
            cutout_dirname = cutout['COADD_OBJECT_ID']
        else:
            cutout_dirname = 'DESJ' + _DecConverter(cutout['RA'], cutout['DEC'])
        cutout_basename = 'DESJ' + _DecConverter(cutout['RA'], cutout['DEC'])
        # Store the sexagecimal representation of the position
        df.at[row_index, 'SEXAGECIMAL'] = cutout_basename
        # Output directory stucture: [base outdir path]/[source tile name]/[position]
        cutout_outdir = os.path.join(outdir, cutout['TILENAME'], cutout_dirname)

        # Make all FITS cutout files necessary for requested FITS files and any RGB files
        all_colors = ''
        fits_colors = ''
        rgb_colors = ''
        # The order matters here, because explicitly requested FITS files may be discarded if the RGB colors are iterated over first
        for rgb_type in [['MAKE_FITS', 'COLORS_FITS'], ['MAKE_RGB_STIFF', 'RGB_STIFF_COLORS'], ['MAKE_RGB_LUPTON', 'RGB_LUPTON_COLORS']]:
            if cutout[rgb_type[0]]:
                # Add the color if it is an acceptable letter do not duplicate
                for color in cutout[rgb_type[1]]:
                    if color in 'grizy' and color not in all_colors:
                        all_colors += color
                        if rgb_type[0] == 'MAKE_FITS':
                            fits_colors += color
                        else:
                            rgb_colors += color
  
        output_fits_files_requested = make_fits_cut(cutout, fits_colors, cutout_outdir, cutout_basename)
        output_fits_files_for_rgb_only = make_fits_cut(cutout, rgb_colors, cutout_outdir, cutout_basename)
        generated_files.extend(output_fits_files_requested)
        all_generated_files.extend(output_fits_files_requested)
        
        if cutout['DISCARD_FITS_FILES']:
            for fits_filename in output_fits_files_for_rgb_only:
                discard_filepaths.append(os.path.join(cutout_outdir, fits_filename))
        else:
            generated_files.extend(output_fits_files_for_rgb_only)
            all_generated_files.extend(output_fits_files_for_rgb_only)

        # Now that all required FITS files have been generated, create any requested RGB images
        for rgb_type in [['MAKE_RGB_STIFF', 'RGB_STIFF_COLORS'], ['MAKE_RGB_LUPTON', 'RGB_LUPTON_COLORS']]:
            if cutout[rgb_type[0]]:
                color_sets = cutout[rgb_type[1]].split(';')
                for color_set in color_sets:
                    output_files = make_rgb(cutout, rgb_type[0], color_set, cutout_outdir, cutout_basename)
                    generated_files.extend(output_files)
                    all_generated_files.extend(output_files)

        # Add new output files to the list of all files generated for this position
        file_list = json.loads(cutout['FILES'])
        df.at[row_index, 'FILES'] = json.dumps(file_list + generated_files)

        # Delete FITS files that were generated only for the purpose of producing RGB images
        if cutout['DISCARD_FITS_FILES']:
            for filepath in discard_filepaths:
                os.remove(filepath)


    # Close database connection
    conn.close()

    # Synchronize parallel processes at this line to ensure all processing is complete
    comm.Barrier()
    # Gather all sub-tables back into one unified table
    gathered_df = comm.gather((df), root=0)
    # Gather the lists of all files generated from parallel processes
    all_generated_files = comm.gather(all_generated_files, root=0)
    if rank == 0:
        # Recombine the slices of the complete DataFrame
        complete_df = pd.concat(gathered_df)
        complete_df.reset_index(drop=True)
        # Save the entire table in the job summary file
        summary['cutouts'] = json.loads(complete_df.to_json(orient="records"))
        # Translate JSON-formatted FILES list to list object type
        for idx, cutout_position in enumerate(summary['cutouts']):
            summary['cutouts'][idx]['FILES'] = json.loads(cutout_position['FILES'])
        # Save the list of unmatched positions
        summary['unmatched_positions'] = {
            'coord': unmatched_coords,
            'coadd': unmatched_coadds,
        }

        logger.info('All processes finished.')
        end2 = time.time()
        processing_time = '{0:.2f}'.format(end2-end1)
        logger.info('Processing took (s): ' + processing_time)
        summary['processing_time'] = processing_time

        # Calculate total size of generated files on disk
        dirsize = getPathSize(outdir)
        dirsize = dirsize * 1. / 1024
        if dirsize > 1024. * 1024:
            dirsize = '{0:.2f} GB'.format(1. * dirsize / 1024. / 1024)
        elif dirsize > 1024.:
            dirsize = '{0:.2f} MB'.format(1. * dirsize / 1024.)
        else:
            dirsize = '{0:.2f} KB'.format(dirsize)
        logger.info('Total file size on disk: {}'.format(dirsize))
        summary['size_on_disk'] = str(dirsize)

        # Record the list of all generated files
        all_generated_files = [y for x in all_generated_files for y in x]
        logger.info('Total number of files: {}'.format(len(all_generated_files)))
        summary['number_of_files'] = len(all_generated_files)

        # Store the job summary info in a JSON-formatted file in a canonical location
        jsonfile = os.path.join(outdir, 'summary.json')
        with open(jsonfile, 'w') as fp:
            json.dump(summary, fp, indent=2)

def validate_config(conf):
    # Cutout positions table must be present
    if 'positions' not in conf or not isinstance(conf['positions'], str):
        msg = 'Invalid cutout positions table'
        logger.error(msg)
        return False, msg
    # User-defined defaults must be valid
    valid, msg = validate_user_defaults(conf)
    if not valid:
        logger.error('Invalid config: {}'.format(msg))
        return False, msg
    # Cutout positions table must be valid
    valid, msg = validate_positions_table(conf['positions'])
    if not valid:
        logger.error('Invalid cutout positions table: {}'.format(msg))
        return False, msg
    return True, ''

def construct_cutouts_table(conf):
    # Load default values
    with open(os.path.join(os.path.dirname(__file__), 'config.default.yaml'), 'r') as configfile:
        defaults = yaml.load(configfile, Loader=yaml.FullLoader)
    # Import CSV-formatted table of positions (and options) to a DataFrame object
    try:
        df = positions_csv_to_dataframe(conf['positions'])
    except Exception as e:
        logger.info('Error importing positions CSV file: {}'.format(str(e).strip()))
        sys.exit(1)

    # Ensure that each parameter column is populated in the DataFrame
    for param in ['xsize', 'ysize', 'colors_fits', 'rgb_stiff_colors', 'rgb_lupton_colors', 'make_fits', 'make_rgb_lupton', 'make_rgb_stiff', 'discard_fits_files']:
        # If the parameter was not included in the CSV file
        if not param in df:
            # Check if a global default was provided by the user.
            if param in conf:
                default_val = conf[param]
            # If not, use the default nominal value
            else:
                default_val = defaults[param]
            df[param] = [default_val for c in range(len(df))]
    
    # Add additional columns required for processing:
    #
    # Add a column marking clearly whether the cutout is based on coordinates or coadd IDs
    df['POSITION_TYPE'] = ['' for c in range(len(df))]
    # Add a column for the adjusted RA value
    df['RA_ADJUSTED'] = [None for c in range(len(df))]
    # Add a column for the output file paths
    df['FILES'] = ['[]' for c in range(len(df))]
    # Add a column for the path to the associated data tile
    df['TILEDIR'] = ['' for c in range(len(df))]
    # Add a column for the name of the associated data tile
    df['TILENAME'] = ['' for c in range(len(df))]
    # Add a column for the sexagecimal representation of the position
    df['SEXAGECIMAL'] = ['' for c in range(len(df))]

    # Iterate over each row and validate parameter values
    for row_index, cutout in df.iterrows():
        # Label the cutout with the type of position to simplify subsequent logic
        if 'coadd_object_id' in cutout and isinstance(cutout['coadd_object_id'], str):
            # The position is based on Coadd ID
            df.at[row_index, 'POSITION_TYPE'] = 'coadd'
        else:
            # The position is based on RA/DEC coordinate
            df.at[row_index, 'POSITION_TYPE'] = 'coord'
            # Round to the desired precision
            coord_significant_digits = abs(round(math.log(COORD_PRECISION, 10.0)))
            ra_val = round(float(cutout['ra']), coord_significant_digits)
            dec_val = round(float(cutout['dec']), coord_significant_digits)
            df.at[row_index, 'ra']  = ra_val
            df.at[row_index, 'dec'] = dec_val
            # Set the adjusted RA value
            df.at[row_index, 'RA_ADJUSTED'] = 360-ra_val if ra_val > 180 else ra_val
        # Set any empty numerical values
        for param in ['xsize', 'ysize', 'rgb_minimum', 'rgb_stretch', 'rgb_asinh']:
            # If the param is not specified or is not a number value
            if not param in cutout or math.isnan(cutout[param]):
                # Check if a global default was provided by the user.
                if param in conf and conf[param] != 0.0:
                    default_val = conf[param]
                # If not, use the default nominal value
                else:
                    default_val = defaults[param]
                df.at[row_index, param] = default_val

        # Ensure that colors are set correctly if output boolean flags are set.
        # The default values are true booleans, but the cutout positions table
        # values are pseudo-boolean numeric values. Translate accordingly.
        for param in ['make_fits', 'make_rgb_stiff', 'make_rgb_lupton', 'discard_fits_files']:
            if not param in cutout or math.isnan(cutout[param]):
                # Check if a global default was provided by the user.
                if param in conf:
                    default_val = 1 if conf[param] == True else 0
                # If not, use the default nominal value
                else:
                    default_val = 1 if defaults[param] == True else 0
                df.at[row_index, param] = default_val
        for param in ['colors_fits', 'rgb_stiff_colors', 'rgb_lupton_colors']:
            if not param in cutout or not isinstance(cutout[param], str):
                # Check if a global default was provided by the user.
                if param in conf:
                    default_val = conf[param]
                # If not, use the default nominal value
                else:
                    default_val = defaults[param]
                df.at[row_index, param] = default_val
    
    # The column heading must be uppercase to allow easier integration with the Oracle DB query results for the tile names
    df = df.rename(str.upper, axis='columns')
    return df

def positions_csv_to_dataframe(positions_csv_text):
    '''Import CSV-formatted table of positions (and options) to a DataFrame object'''
    df = None
    try:
        positions_list = positions_csv_text.split('\n')
        positions_list[0] = positions_list[0].lower()
        positions_csv_text_lowercase_column_headings = '\n'.join(positions_list)
        df = pd.DataFrame(pd.read_csv(StringIO(positions_csv_text_lowercase_column_headings), skipinitialspace=True, dtype={
            'coadd_object_id': str,
            'ra': np.float64,
            'dec': np.float64,
            'xsize': np.float64,
            'ysize': np.float64,
            'colors_fits': str,
            'rgb_stiff_colors': str,
            'rgb_lupton_colors': str,
            # The boolean values are represented as float, where zero == False and non-zero == True.
            # If 0 or 1 (no-zero) is specified, it overrides the user-defined default value. If nothing is specified,
            # the NaN value indicates to use the user-defined default value for that cutout position.
            'make_fits': np.float64,
            'make_rgb_stiff': np.float64,
            'make_rgb_lupton': np.float64,
            'discard_fits_files': np.float64,
            'rgb_minimum': np.float64,
            'rgb_stretch': np.float64,
            'rgb_asinh': np.float64,
        },
        na_values={
            'coadd_object_id': '',
            'ra': '',
            'dec': '',
            'xsize': '',
            'ysize': '',
            'colors_fits': '',
            'rgb_stiff_colors': '',
            'rgb_lupton_colors': '',
            'make_fits': '',
            'make_rgb_stiff': '',
            'make_rgb_lupton': '',
            'discard_fits_files': '',
            'rgb_minimum': '',
            'rgb_stretch': '',
            'rgb_asinh': '',
        }))
    except Exception as e:
        status = STATUS_ERROR
        msg = str(e).strip()
        return status, msg
    return df

def is_valid_color_band(color):
    return isinstance(color, str) and len(color) == 1 and color.lower() in 'grizy'

def valid_fits_colors(colors_string):
    '''A FITS color set is a string of color bands with at least one color band specified'''
    # Ignore case of letters
    colors_string = colors_string.lower()
    if not colors_string or not isinstance(colors_string, str):
        return False
    valid_colors = ''
    for color in colors_string:
        # Detect invalid color bands
        if not is_valid_color_band(color):
            return False
        # Detect redundant colors
        if color in valid_colors:
            return False
        # Append valid color to redundancy checking string
        valid_colors += color
    return True

def valid_rgb_color_set(colors_string):
    '''An RGB color set is a three-character string of color bands'''
    # Ignore case of letters
    colors_string = colors_string.lower()
    if not isinstance(colors_string, str):
        return False
    # There must be exactly three colors
    if len(colors_string) != 3:
        return False
    valid_colors = ''
    for color in colors_string:
        # Detect redundant colors
        if color in valid_colors:
            return False
        # Detect invalid color bands
        if not is_valid_color_band(color):
            return False
        # Append valid color to redundancy checking string
        valid_colors += color
    return True

def valid_rgb_color_sets(color_set_string):
    '''The color sets are specified in a semi-colon-delineated string of color band triplets'''
    if not isinstance(color_set_string, str):
        return False
    valid_color_sets = []
    for color_set in color_set_string.split(';'):
        # Detect redundant color sets
        if color_set in valid_color_sets:
            return False
        # Detect invalid color sets
        if not valid_rgb_color_set(color_set):
            return False
        valid_color_sets.append(color_set)
    return True

def validate_user_defaults(conf):
    msg = ''
    if not all(k in conf for k in ['username', 'password']) or not conf['username'] or not conf['password']:
        msg = 'A valid username and password must be provided.'
        return False, msg
    if 'jobid' in conf and not isinstance(conf['jobid'], str):
        msg = 'jobid must be a string if specified.'
        return False, msg

    # Load the default values
    with open(os.path.join(os.path.dirname(__file__), 'config.default.yaml'), 'r') as configfile:
        defaults = yaml.load(configfile, Loader=yaml.FullLoader)

    # Validate selected database
    if 'db' not in conf or conf['db'].upper() not in VALID_DATA_SOURCES:
        msg = 'Please select a valid database: {}.'.format(VALID_DATA_SOURCES)
        return False, msg
    if 'release' not in conf or conf['release'].upper() not in VALID_DATA_SOURCES[conf['db'].upper()]:
        msg = 'Please select a valid data release: {}.'.format(VALID_DATA_SOURCES[conf['db'].upper()])
        return False, msg

    # Validate default cutout size dimensions
    for param in ['xsize', 'ysize']:
        if param in conf:
            if not isinstance(conf[param], (int, float)):
                return False, 'Cutout size dimensions (if supplied) must be numeric values'
            if conf[param] < defaults['{}_min'.format(param)]:
                return False, 'Cutout size minimum value is {}'.format(defaults['{}_min'.format(param)])
            if conf[param] > defaults['{}_max'.format(param)]:
                return False, 'Cutout size maximum value is {}'.format(defaults['{}_max'.format(param)])

    # Validate default boolean flags for which images to produce
    for param in ['make_fits', 'make_rgb_lupton', 'make_rgb_stiff', 'discard_fits_files']:
        if param in conf:
            if not isinstance(conf[param], bool):
                return False, '"{}" must be a boolean value'.format(param)
    # Validate default FITS color bands
    for param in ['colors_fits']:
        if param in conf:
            if not valid_fits_colors(conf[param]):
                return False, 'Invalid FITS color set specified: "{}" '.format(param)
    # Validate default RGB color sets
    for param in ['rgb_stiff_colors', 'rgb_lupton_colors']:
        if param in conf:
            if not valid_rgb_color_sets(conf[param]):
                return False, 'Invalid RGB color set specified: "{}" '.format(param)
    # Validate default Lupton parameters
    for param in ['rgb_minimum', 'rgb_stretch', 'rgb_asinh']:
        if param in conf:
            if not isinstance(conf[param], (int, float)):
                return False, '"{}" must have a numeric value'.format(param)
    return True, msg

def validate_positions_table(positions_csv_text):
    msg = ''
    valid_column_headers = [
        'ra',
        'dec',
        'coadd_object_id',
        'xsize',
        'ysize',
        'colors_fits',
        'rgb_stiff_colors',
        'rgb_lupton_colors',
        'rgb_minimum',
        'rgb_stretch',
        'rgb_asinh',
        'make_fits',
        'make_rgb_stiff',
        'make_rgb_lupton',
        'discard_fits_files',
    ]
    try:
        # Import the table as a DataFrame
        df = positions_csv_to_dataframe(positions_csv_text)
        # Load the default values
        with open(os.path.join(os.path.dirname(__file__), 'config.default.yaml'), 'r') as configfile:
            defaults = yaml.load(configfile, Loader=yaml.FullLoader)
    except Exception as e:
        msg = str(e).strip()
        return False, msg
    # Check for invalid column headers
    invalid_column_headers = []
    for column_header in df.columns:
        if column_header.lower() not in valid_column_headers:
            invalid_column_headers.append(column_header)
    if invalid_column_headers:
        return False, 'Invalid columns in positions table: {}'.format(invalid_column_headers)
    # Iterate over each position
    for row_index, cutout in df.iterrows():
        valid_coadd_id = valid_coords = False
        if 'coadd_object_id' in cutout and isinstance(cutout['coadd_object_id'], str):
            valid_coadd_id = True
        if 'ra' in cutout and not math.isnan(cutout['ra']) and 'dec' in cutout and not math.isnan(cutout['dec']):
            valid_coords = True
        # If both COADD ID and RA/DEC coords are specified, fail
        if valid_coords and valid_coadd_id:
            return False, 'Only COADD_OBJECT_ID or RA/DEC coordinates my be specified, not both.'
        # Either RA/DEC or Coadd ID must be specified
        if not valid_coords and not valid_coadd_id:
            return False, 'Each row must have either RA/DEC or COADD_OBJECT_ID'
    
        # Ensure numerical values with limited ranges are respected
        for param in ['xsize', 'ysize', 'rgb_minimum', 'rgb_stretch', 'rgb_asinh']:
            param_min = '{}_min'.format(param)
            param_max = '{}_max'.format(param)
            # If the param is not specified or is not a number value
            if param in cutout and not math.isnan(cutout[param]):
                if not isinstance(cutout[param], (int, float)):
                    return False, 'Invalid parameter value for "{}"'.format(param)
                # If the size is too small, rail at minimum, if there is a minimum
                if param_min in defaults and cutout[param] < defaults[param_min]:
                    return False, 'Parameter "{}" is too small: minimum valid value is "{}"'.format(param, param_min)
                # If the size is too large, rail at maximum, if there is a maximum
                if param_max in defaults and cutout[param] > defaults[param_max]:
                    return False, 'Parameter "{}" is too large: maximum valid value is "{}"'.format(param, param_max)
        # Validate boolean flags for which images to produce
        for param in ['make_fits', 'make_rgb_lupton', 'make_rgb_stiff', 'discard_fits_files']:
            if param in cutout:
                if not math.isnan(cutout[param]) and not isinstance(cutout[param], (int, float)) :
                    return False, '"{}" must have value 0 (disabled) or 1 (enabled) if provided in the cutout positions table.'.format(param)
        # Validate color strings
        if 'colors_fits' in cutout and isinstance(cutout['colors_fits'], str) and len(cutout['colors_fits']) > 0:
            if not valid_fits_colors(cutout['colors_fits']):
                return False, 'Invalid FITS colors'
        if 'rgb_stiff_colors' in cutout and isinstance(cutout['rgb_stiff_colors'], str) and len(cutout['rgb_stiff_colors']) > 0:
            if not valid_rgb_color_sets(cutout['rgb_stiff_colors']):
                return False, 'Invalid STIFF RGB colors'
        if 'rgb_lupton_colors' in cutout and isinstance(cutout['rgb_lupton_colors'], str) and len(cutout['rgb_lupton_colors']) > 0:
            if not valid_rgb_color_sets(cutout['rgb_lupton_colors']):
                return False, 'Invalid Lupton RGB colors'
    return True, msg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This program will make any number of cutouts, using the master tiles.")

    # Config file
    parser.add_argument('--config', type=str, required=True, help='YAML-formatted configuration file specifying cutout positions and options.')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    # Load the configuration file from disk
    with open(args.config, 'r') as configfile:
        conf = yaml.load(configfile, Loader=yaml.FullLoader)

    run(conf)
