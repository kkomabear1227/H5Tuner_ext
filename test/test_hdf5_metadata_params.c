/*
 * Copyright by The HDF Group.
 * All rights reserved.
 *
 * This file is part of h5tuner. The full h5tuner copyright notice,
 * including terms governing use, modification, and redistribution, is
 * contained in the file COPYING, which can be found at the root of the
 * source code distribution tree.  If you do not have access to this file,
 * you may request a copy from help@hdfgroup.org.
 */

/*
 * Checks the five HDF5 parameters added to the shim on 2026-08-13:
 *
 *     meta_block_size       H5Pset_meta_block_size
 *     chunk_cache           H5Pset_cache, rdcc_nbytes only
 *     coll_metadata_write   H5Pset_coll_metadata_write      (HDF5 1.10+)
 *     col_meta_ops          H5Pset_all_coll_metadata_ops    (HDF5 1.10+)
 *     mdc_conf              H5Pset_mdc_config, named presets
 *
 * Expected values come from test/config.xml.in.
 *
 * Deliberately not written like the other tests here.  Those are fourteen
 * copies of ph5example.c -- 16,800 lines carrying about 600 lines of unique
 * logic -- because each one needed an independent pass/fail signal from
 * Automake.  All five parameters below are injected at the same moment, in
 * H5Fcreate, so one program can check all of them and a real parallel write
 * workload is not needed to observe the injection.  Under 200 lines instead of
 * 1,300.
 *
 * What this does NOT verify: that the values reach the filesystem or change
 * anything.  It reads the property list back, which shows the shim wrote to it.
 * That is the same limitation the other tests have -- see
 * 03-원본-코드-분석.md 7.3.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mpi.h"
#include "hdf5.h"

#define TEST_FILE "metadata_params.h5"

/* Expected values, matching test/config.xml.in. */
#define EXPECT_META_BLOCK_SIZE   262144
#define EXPECT_CHUNK_CACHE     16777216
#define EXPECT_COLL_META_WRITE        1
#define EXPECT_COL_META_OPS           1
#define EXPECT_MDC_PRESET     "aggressive"
/* Sizes the shim's "aggressive" preset installs. */
#define EXPECT_MDC_MIN_SIZE     (4 * 1024 * 1024)
#define EXPECT_MDC_INITIAL     (32 * 1024 * 1024)
#define EXPECT_MDC_MAX_SIZE   (128 * 1024 * 1024)

static int nerrors = 0;
static int verbose = 1;

static void report(const char *label, int ok, const char *detail)
{
    if (ok) {
        if (verbose)
            printf("PASSED: %-24s %s\n", label, detail);
    } else {
        printf("FAILED: %-24s %s\n", label, detail);
        nerrors++;
    }
}

int main(int argc, char **argv)
{
    hid_t fapl, file;
    herr_t ret;
    int rank, size;
    char detail[256];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    verbose = (rank == 0);

    if (verbose) {
        printf("\n");
        printf("==================================================\n");
        printf("H5Tuner: HDF5 metadata/cache parameter injection\n");
        printf("==================================================\n");
        printf("LD_PRELOAD=%s\n", getenv("LD_PRELOAD") ?
               getenv("LD_PRELOAD") : "(unset)");
        printf("HDF5 %d.%d.%d, %d process(es)\n",
               H5_VERS_MAJOR, H5_VERS_MINOR, H5_VERS_RELEASE, size);
    }

    /*
     * The shim refuses any driver other than MPI-IO and returns -1 without
     * calling through to the real H5Fcreate, so the MPI-IO driver has to be set
     * before the file is created or nothing works.
     */
    fapl = H5Pcreate(H5P_FILE_ACCESS);
    assert(fapl != FAIL);
    ret = H5Pset_fapl_mpio(fapl, MPI_COMM_WORLD, MPI_INFO_NULL);
    assert(ret != FAIL);

    /* This is the call the shim intercepts.  Everything below reads back what
     * it did to fapl. */
    file = H5Fcreate(TEST_FILE, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    if (file == FAIL) {
        printf("FAILED: H5Fcreate returned -1.  Either LD_PRELOAD does not "
               "point at libautotuner.so, or config.xml is missing from the "
               "working directory.\n");
        nerrors++;
        goto done;
    }

    /* -- meta_block_size ------------------------------------------------- */
    {
        hsize_t block = 0;
        ret = H5Pget_meta_block_size(fapl, &block);
        assert(ret != FAIL);
        snprintf(detail, sizeof(detail), "expected %d, got %llu",
                 EXPECT_META_BLOCK_SIZE, (unsigned long long) block);
        report("meta_block_size",
               (unsigned long long) block == EXPECT_META_BLOCK_SIZE, detail);
    }

    /* -- chunk_cache ----------------------------------------------------- */
    {
        int mdc_nelmts = 0;
        size_t nslots = 0, nbytes = 0;
        double w0 = 0.0;
        ret = H5Pget_cache(fapl, &mdc_nelmts, &nslots, &nbytes, &w0);
        assert(ret != FAIL);
        snprintf(detail, sizeof(detail),
                 "expected rdcc_nbytes %d, got %llu (nslots %llu, w0 %.2f)",
                 EXPECT_CHUNK_CACHE, (unsigned long long) nbytes,
                 (unsigned long long) nslots, w0);
        report("chunk_cache",
               (unsigned long long) nbytes == EXPECT_CHUNK_CACHE, detail);
    }

    /* -- coll_metadata_write, col_meta_ops ------------------------------- */
#if H5_VERSION_GE(1, 10, 0)
    {
        hbool_t collective = 0;
        ret = H5Pget_coll_metadata_write(fapl, &collective);
        assert(ret != FAIL);
        snprintf(detail, sizeof(detail), "expected %d, got %d",
                 EXPECT_COLL_META_WRITE, (int) collective);
        report("coll_metadata_write",
               (int) collective == EXPECT_COLL_META_WRITE, detail);

        collective = 0;
        ret = H5Pget_all_coll_metadata_ops(fapl, &collective);
        assert(ret != FAIL);
        snprintf(detail, sizeof(detail), "expected %d, got %d",
                 EXPECT_COL_META_OPS, (int) collective);
        report("col_meta_ops",
               (int) collective == EXPECT_COL_META_OPS, detail);
    }
#else
    if (verbose)
        printf("SKIPPED: coll_metadata_write and col_meta_ops need HDF5 "
               "1.10.0 or later; this build is %d.%d.%d\n",
               H5_VERS_MAJOR, H5_VERS_MINOR, H5_VERS_RELEASE);
#endif

    /* -- mdc_conf -------------------------------------------------------- */
    {
        H5AC_cache_config_t mdc;
        mdc.version = H5AC__CURR_CACHE_CONFIG_VERSION;
        ret = H5Pget_mdc_config(fapl, &mdc);
        assert(ret != FAIL);
        snprintf(detail, sizeof(detail),
                 "preset %s: min %llu initial %llu max %llu",
                 EXPECT_MDC_PRESET,
                 (unsigned long long) mdc.min_size,
                 (unsigned long long) mdc.initial_size,
                 (unsigned long long) mdc.max_size);
        report("mdc_conf",
               (unsigned long long) mdc.min_size == EXPECT_MDC_MIN_SIZE &&
               (unsigned long long) mdc.initial_size == EXPECT_MDC_INITIAL &&
               (unsigned long long) mdc.max_size == EXPECT_MDC_MAX_SIZE,
               detail);
    }

    H5Fclose(file);
    /* H5Fclose is collective, but the barrier makes the ordering explicit
     * rather than relying on that. */
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        remove(TEST_FILE);

done:
    H5Pclose(fapl);

    if (verbose) {
        printf("--------------------------------------------------\n");
        if (nerrors)
            printf("%d parameter(s) did not reach the property list.\n",
                   nerrors);
        else
            printf("All checked parameters were injected.\n");
        printf("--------------------------------------------------\n\n");
    }

    MPI_Finalize();
    return nerrors ? 1 : 0;
}
