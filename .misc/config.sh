#!/bin/bash

MPICC=$(command -v mpicc)
if [ ! -z $MPICC ]; then
     # using mpich available on the system
     echo "--"
     echo "using mpicc:  ${MPICC} ";
     echo "--"

# else is at bottom of the script

# Assume AT_DESTDIR as path to AT directory
AT_PREFIX=${PWD}

if [ ! -d $AT_PREFIX/opt ]
then
        echo "--"
        echo "Creating ${AT_PREFIX}/opt and building AT"
        echo "--"

        mkdir -p $AT_PREFIX/opt
fi

# HDF5 1.14.1-2 is what the target cluster has installed (parallel ON,
# Fortran/C++ bindings OFF).  The version matters to src/autotuner_hdf5.c:
# H5Pset_coll_metadata_write and H5Pset_all_coll_metadata_ops arrived in
# 1.10.0, and the shim compiles them out below that with a runtime warning.
HDF5_V="hdf5-1.14.1-2"
# Upstream release tarballs use underscores in the version, directories use dots.
HDF5_TAG="hdf5-1_14_1-2"
HDF5_SERIES="hdf5-1.14"
HDF5_POINT="hdf5-1.14.1"
ZLIB_V="zlib-1.2.8"
MXML_V="mxml-2.9"
NCDF4_V="v4.4.0"

HDF5DIR="${HDF5_V}"
ZLIBDIR="${ZLIB_V}"
MXMLDIR="${MXML_V}"
NCDF4DIR="netcdf-c-4.4.0"

if [ ! -d $MXMLDIR ]
then
        echo "--";
        echo "Creating ${MXMLDIR} and Building Mini-XML"
        echo "--";

        # msweet.org no longer serves project3.  VERIFY before relying on this.
        wget https://github.com/michaelrsweet/mxml/releases/download/release-2.9/$MXML_V.tar.gz \
            || wget http://www.msweet.org/files/project3/$MXML_V.tar.gz
        tar xvzf $MXML_V.tar.gz
        cd $MXMLDIR
            ./configure --prefix=${AT_PREFIX}/opt/${MXML_V} --enable-shared
            make && make install
        cd ..
        # Remove download file
        rm -f "${MXML_V}.tar.gz"
else
        echo "A directory " $MXMLDIR " already exists.";
fi

if [ ! -d $ZLIBDIR ]
then
        echo "--";
        echo "Creating ${ZLIBDIR} and Building Zlib"
        echo "--";

        wget http://zlib.net/${ZLIB_V}.tar.gz
        tar xvzf ${ZLIB_V}.tar.gz
        cd ${ZLIBDIR}
            ./configure --prefix=${AT_PREFIX}/opt/${ZLIB_V}
             make && make install
        cd ..
        # Remove download file
        rm -f "${ZLIB_V}.tar.gz"
else
        echo "--";
        echo "A directory " $ZLIBDIR " already exists.";
        echo "--";
fi


if [ ! -d $HDF5DIR ]
then
        echo "--";
        echo "Creating ${HDF5DIR} and Building HDF5"
        echo "--";

        # The hdfgroup.org FTP layout this script was written against is gone.
        # Try the GitHub release first, then the support site.  VERIFY THESE
        # URLS before relying on the bootstrap -- they have moved twice already.
        wget https://github.com/HDFGroup/hdf5/releases/download/${HDF5_TAG}/${HDF5_TAG}.tar.gz \
            || wget https://support.hdfgroup.org/ftp/HDF5/releases/${HDF5_SERIES}/${HDF5_POINT}/src/${HDF5_V}.tar.gz
        if [ -f "${HDF5_TAG}.tar.gz" ]; then
            tar xvzf ${HDF5_TAG}.tar.gz
        else
            tar xvzf ${HDF5_V}.tar.gz
        fi

        cd ${HDF5DIR}
            if [ -z "$MPIROOT" ]; then
                CC=mpicc \
                ./configure --prefix=${AT_PREFIX}/opt/${HDF5_V}/ --enable-parallel  --enable-shared \
                --with-zlib=${AT_PREFIX}/opt/${ZLIB_V}
            else
                export PATH=/opt/mpich2/bin:$PATH
                export LD_LIBRARY_PATH=/opt/mpich2/lib:$LD_LIBRARY_PATH

            fi
            make && make install
        cd ..
        # Remove downloaded file
        rm -f "${HDF5_V}.tar.gz" "${HDF5_TAG}.tar.gz"
else
        echo "--";
        echo "A directory " $HDF5DIR " already exists.";
        echo "--";
fi

# Build NetCDF4 if AT_NCDF4 is defined
#AT_NCDF4="BUILD"

if [ ! -d $NCDF4DIR ] && [ ! -z $AT_NCDF4 ]
then
        echo "--";
        echo "Creating ${NCDF4DIR} and Building NetCD4"
        echo "--";

        wget https://github.com/Unidata/netcdf-c/archive/${NCDF4_V}.tar.gz
        tar xvzf ${NCDF4_V}.tar.gz

        cd ${NCDF4DIR}
            export LD_LIBRARY_PATH="${AT_PREFIX}/opt/${HDF5_V}/lib":"$LD_LIBRARY_PATH"
            CC=mpicc \
            CPPFLAGS=-I${AT_PREFIX}/opt/${HDF5_V}/include \
            LDFLAGS=-L${AT_PREFIX}/opt/${HDF5_V}/lib \
            ./configure --enable-parallel-tests \
            --prefix=${AT_PREFIX}/opt/${NCDF4DIR}
            make && make install
        cd ..
        # Remove downloaded file
        rm -f "${NCDF4_V}.tar.gz"
# Pre-existing bug: this fi was missing, so the else at the bottom of the script
# attached to the NetCDF4 block instead of the mpicc check at the top and the
# whole file failed `bash -n`.  It has never been syntactically valid.
fi


# checking for mpich -- continuing from the top of the script
else
   # no mpich notify and exit
   echo "Could not find mpicc. Please specify the mpich being used. Aborting.";
fi
