/* Minimal HDF5 declarations, for parsing only.
 *
 * The I/O discovery component runs clang over application source, and clang
 * cannot build an AST for a call whose types it does not know: a statement
 * like `hid_t f = H5Fcreate(...)` disappears from the tree entirely when
 * hid_t is undeclared, taking the I/O call with it.  On the target machine
 * the real headers are passed with --clang-arg -I$HDF5ROOT/include; this stub
 * exists so the component can be developed and tested on a machine with no
 * HDF5 installed.
 *
 * Declarations only.  Nothing here is ABI-compatible and nothing links.
 */

#ifndef AUTOTUNER_STUB_HDF5_H
#define AUTOTUNER_STUB_HDF5_H

#include <stddef.h>

typedef int hid_t;
typedef int herr_t;
typedef unsigned long long hsize_t;
typedef long long hssize_t;
typedef int htri_t;

#define H5P_DEFAULT 0
#define H5F_ACC_TRUNC 0x0002u
#define H5F_ACC_RDWR 0x0001u
#define H5F_ACC_RDONLY 0x0000u
#define H5S_ALL 0
#define H5T_NATIVE_FLOAT 1
#define H5T_NATIVE_DOUBLE 2
#define H5T_NATIVE_INT 3
#define H5T_NATIVE_CHAR 4
#define H5P_FILE_ACCESS 5
#define H5P_DATASET_XFER 6
#define H5P_DATASET_CREATE 7
#define H5S_SELECT_SET 0
#define H5FD_MPIO_COLLECTIVE 1
#define H5FD_MPIO_INDEPENDENT 0

hid_t H5Fcreate(const char *name, unsigned flags, hid_t fcpl, hid_t fapl);
hid_t H5Fopen(const char *name, unsigned flags, hid_t fapl);
herr_t H5Fflush(hid_t obj, int scope);
herr_t H5Fclose(hid_t file);

hid_t H5Screate_simple(int rank, const hsize_t *dims, const hsize_t *max);
herr_t H5Sselect_hyperslab(hid_t space, int op, const hsize_t *start,
                           const hsize_t *stride, const hsize_t *count,
                           const hsize_t *block);
herr_t H5Sclose(hid_t space);

hid_t H5Dcreate2(hid_t loc, const char *name, hid_t type, hid_t space,
                 hid_t lcpl, hid_t dcpl, hid_t dapl);
hid_t H5Dcreate(hid_t loc, const char *name, hid_t type, hid_t space,
                hid_t dcpl);
hid_t H5Dopen2(hid_t loc, const char *name, hid_t dapl);
herr_t H5Dwrite(hid_t dset, hid_t mem_type, hid_t mem_space,
                hid_t file_space, hid_t dxpl, const void *buf);
herr_t H5Dread(hid_t dset, hid_t mem_type, hid_t mem_space,
               hid_t file_space, hid_t dxpl, void *buf);
herr_t H5Dclose(hid_t dset);

hid_t H5Gcreate2(hid_t loc, const char *name, hid_t lcpl, hid_t gcpl,
                 hid_t gapl);
herr_t H5Gclose(hid_t group);

hid_t H5Pcreate(hid_t cls);
herr_t H5Pset_alignment(hid_t fapl, hsize_t threshold, hsize_t alignment);
herr_t H5Pset_sieve_buf_size(hid_t fapl, size_t size);
herr_t H5Pset_meta_block_size(hid_t fapl, hsize_t size);
herr_t H5Pset_chunk(hid_t dcpl, int ndims, const hsize_t *dim);
herr_t H5Pset_dxpl_mpio(hid_t dxpl, int mode);
herr_t H5Pclose(hid_t plist);

#endif /* AUTOTUNER_STUB_HDF5_H */
