
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


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <libgen.h>
#include <pwd.h>
#include "mpi.h"
#include "hdf5.h"

/*
 * HDF5 1.12+ defines the async entry points as variadic macros that inject
 * __FILE__, __func__ and __LINE__ ahead of the caller's arguments:
 *
 *   #define H5Fcreate_async(...) H5Fcreate_async(__FILE__, __func__, ...)
 *
 * That is right for callers and wrong for us: we are implementing the
 * functions those macros expand to, so the macro has to be out of the way
 * before the definitions and before dlsym() looks the names up.
 */
#ifdef H5Fcreate_async
#undef H5Fcreate_async
#endif
#ifdef H5Dcreate_async
#undef H5Dcreate_async
#endif
#ifdef H5Dwrite_async
#undef H5Dwrite_async
#endif

#include "autotuner.h"
#include "mxml.h"

/*
typedef int hid_t;
typedef int herr_t;
*/

#define __USE_GNU
#include <dlfcn.h>
#include <stdlib.h>

#define FORWARD_DECL(name, ret, args) \
    ret (*__fake_ ## name)args = NULL;

#define DECL(__name) __name

#define MAP_OR_FAIL(func) \
	if(!(__fake_ ## func)) \
	{ \
		__fake_ ## func = dlsym(RTLD_NEXT, #func); \
		if(!(__fake_ ## func)) { \
			fprintf(stderr, "H5Tuner failed to map symbol: %s\n", #func); \
			exit(1); \
		} \
	}

/*
 * Loads the configuration document, or returns NULL.
 *
 * Every hook had its own copy of this, and every copy had the same three
 * defects: the path hardcoded to the process working directory, fopen unchecked
 * before its stream went to the XML parser, and neither the stream nor the
 * parsed document ever released.  H5Dcreate runs once per dataset, so the leak
 * there was worse than in H5Fcreate.
 *
 * AT_CONFIG_FILE was present in the original and commented out.  With it the
 * path can be set per rank or per candidate, which is what lets a tuner
 * evaluate configurations concurrently instead of sharing one ./config.xml.
 *
 * The caller owns the returned document and must mxmlDelete() it.
 */
static mxml_node_t *at_load_config(const char *context)
{
	char file_path[1024];
	const char *config_file;
	FILE *fp;
	mxml_node_t *tree;

	config_file = getenv("AT_CONFIG_FILE");
	if(config_file != NULL && config_file[0] != '\0') {
		strncpy(file_path, config_file, sizeof(file_path) - 1);
		file_path[sizeof(file_path) - 1] = '\0';
	}
	else {
		strcpy(file_path, "config.xml");
	}

	fp = fopen(file_path, "r");
	if(fp == NULL) {
		fprintf(stderr, "H5Tuner: cannot open %s; %s proceeds untuned\n",
		        file_path, context);
		return NULL;
	}

	tree = mxmlLoadFile(NULL, fp, MXML_TEXT_CALLBACK);
	fclose(fp);
	if(tree == NULL) {
		fprintf(stderr, "H5Tuner: cannot parse %s; %s proceeds untuned\n",
		        file_path, context);
	}
	return tree;
}

/*
 * GPFS lockless I/O is requested by prefixing the file name with "bglockless:",
 * so unlike every other parameter here it changes the name handed to HDF5
 * rather than a property list.
 *
 * The previous version could not do that.  It built the prefixed name in a
 * local buffer and then did strcpy(filename, new_filename) into the caller's
 * string -- which is const, is eleven bytes shorter than what was being
 * written, and was never seen again because H5Fcreate went on to pass its own
 * unmodified `filename` to the real call.  So IBM_lockless_io either did
 * nothing or overflowed the application's buffer, depending on where the name
 * came from.  The malloc'd buffer leaked either way.
 *
 * Now the prefixed name goes into a caller-provided buffer and the caller
 * decides what to open.  Returns 1 when the parameter applies to this file.
 */
static int gpfs_lockless_filename(mxml_node_t *tree, const char *filename,
                                  char *out, size_t out_size)
{
	static const char prefix[] = "bglockless:";
	const char *node_file_name;
	mxml_node_t *node;

	for(node = mxmlFindElement(tree, tree, "IBM_lockless_io", NULL, NULL, MXML_DESCEND);
	    node != NULL;
	    node = mxmlFindElement(node, tree, "IBM_lockless_io", NULL, NULL, MXML_DESCEND)) {
		node_file_name = mxmlElementGetAttr(node, "FileName");
		// TODO: Change this strstr() function call with a use of basename!
		if((node_file_name != NULL) && (strstr(filename, node_file_name) == NULL))
			continue;
		if(node->child == NULL)
			continue;
		if(strcmp(node->child->value.text.string, "true") != 0)
			continue;

		if(strlen(prefix) + strlen(filename) + 1 > out_size) {
			fprintf(stderr, "H5Tuner: \"%s%s\" does not fit in %lu bytes; "
			        "IBM_lockless_io not applied\n", prefix, filename,
			        (unsigned long) out_size);
			return 0;
		}
		strcpy(out, prefix);
		strcat(out, filename);
		return 1;
	}
	return 0;
}

void set_mpi_parameter(mxml_node_t *tree, char *parameter_name, const char *filename, MPI_Info *orig_info)
{
	const char *node_file_name;
	mxml_node_t *node;

	for(node = mxmlFindElement(tree, tree, parameter_name, NULL, NULL,MXML_DESCEND); node != NULL; node = mxmlFindElement(node, tree, parameter_name, NULL, NULL,MXML_DESCEND)) {
		node_file_name = mxmlElementGetAttr(node, "FileName");
		#ifdef DEBUG
		//printf("Node_file_name: %s\n", node_file_name);
		#endif
		// TODO: Change this strstr() function call with a use of basename!
		if(node_file_name == NULL)  {
			#ifdef DEBUG
			//printf("H5Tuner: Execution wide setting %s: %s\n", parameter_name, node->child->value.text.string);
			#endif
			MPI_Info_set(*orig_info, parameter_name, node->child->value.text.string);
		}
		else {
      if( strstr(filename, node_file_name) != NULL )  {
        #ifdef DEBUG
        //printf("H5Tuner: %s setting %s: %s\n\n\n",node_file_name, parameter_name, node->child->value.text.string);
        #endif
        MPI_Info_set(*orig_info, parameter_name, node->child->value.text.string);
      }
			//continue;
		}
	}
}

#if H5_VERSION_GE(1,10,0)
/*
 * Boolean values in config.xml.  The tuner writes 0 or 1; a hand-edited file is
 * just as likely to say true/yes, so both are accepted.  Anything else is false.
 */
static hbool_t at_parse_bool(const char *text)
{
	if(text == NULL)
		return 0;
	while(*text == ' ' || *text == '\t')
		text++;
	if(*text == '1' || *text == 't' || *text == 'T' || *text == 'y' || *text == 'Y')
		return 1;
	return 0;
}
#endif

hid_t set_hdf5_parameter(mxml_node_t *tree, char *parameter_name, const char *filename, hid_t fapl_id)
{
	const char *node_file_name;
	mxml_node_t *node;
	hid_t hasChunk=-1;

	for(node = mxmlFindElement(tree, tree, parameter_name, NULL, NULL,MXML_DESCEND); node != NULL; node = mxmlFindElement(node, tree, parameter_name, NULL, NULL,MXML_DESCEND)) {
		node_file_name = mxmlElementGetAttr(node, "FileName");
		#ifdef DEBUG
		  //printf("Node_file_name: %s\n", node_file_name);
		#endif
		// TODO: Change this strstr() function call with a use of basename!
		if((node_file_name == NULL) || (strstr(filename, node_file_name) != NULL))  {
			/*
			 * An element with no text child would dereference NULL in every
			 * branch below.  The static shim checks this; the dynamic one did
			 * not, so an empty <sieve_buf_size/> crashed the application.
			 */
			if(node->child == NULL) {
				fprintf(stderr, "H5Tuner: <%s> in config.xml has no value; skipping\n",
				        parameter_name);
				continue;
			}
				#ifdef DEBUG
				  //printf("H5Tuner: setting %s: %s\n", parameter_name, node->child->value.text.string);
				#endif
			if(strcmp(parameter_name, "sieve_buf_size") == 0) {
				if(H5Pset_sieve_buf_size(fapl_id,
				        (size_t) strtoull(node->child->value.text.string, NULL, 10)) < 0) {
					fprintf(stderr, "H5Tuner: H5Pset_sieve_buf_size(%s) failed\n",
					        node->child->value.text.string);
				}
			}
			else if(strcmp(parameter_name, "alignment") == 0) {
				char *threshold = strtok(node->child->value.text.string, ",");
				char *alignment = strtok(NULL, ",");
				#ifdef DEBUG
				  //printf("H5Tuner: setting Threshold=%s; Alignment=%s\n", threshold, alignment);
				#endif

				if(H5Pset_alignment(fapl_id,
				        (hsize_t) strtoull(threshold, NULL, 10),
				        (hsize_t) strtoull(alignment, NULL, 10)) < 0) {
					fprintf(stderr, "H5Tuner: H5Pset_alignment(%s, %s) failed\n",
					        threshold, alignment);
				}
			}
			else if(strcmp(parameter_name, "chunk") == 0) {
				const char* variable_name = mxmlElementGetAttr(node, "VariableName");
				#ifdef DEBUG
				  //printf("H5Tuner: VariableName: %s\n", variable_name);
				#endif

				if(variable_name == NULL || (variable_name != NULL && strcmp(variable_name, filename) == 0)) {

				    int ndims = H5Sget_simple_extent_ndims(fapl_id);
				    hsize_t *dims = (hsize_t *) malloc(sizeof(hsize_t) * ndims);
				    H5Sget_simple_extent_dims(fapl_id, dims, NULL);

				    #ifdef DEBUG
				      //printf("dims[0] = %d, ndims = %d\n", dims[0], ndims);
				      //printf("dims[1] = %d, ndims = %d\n", dims[1], ndims);
				      //printf("dims[2] = %d, ndims = %d\n", dims[2], ndims);
				    #endif

				    hsize_t *chunk_arr = (hsize_t *) malloc(sizeof(hsize_t) * ndims);
				    int i;
				    chunk_arr[0] = atoi(strtok(node->child->value.text.string, ","));
				    if(chunk_arr[0] > dims[0])
					    return 0;
				    #ifdef DEBUG
				      //printf("H5Tuner: Setting chunk[0] for %s -> %d\n", filename, chunk_arr[0]);
				    #endif
				    for(i = 1; i < ndims; i++) {
					    chunk_arr[i] = atoi(strtok(NULL, ","));
				    	if(chunk_arr[i] > dims[i])
					      return 0;
				    	#ifdef DEBUG
				    	  //printf("H5Tuner: Setting chunk[%d] for %s -> %d\n", i, filename, chunk_arr[i]);
				    	#endif
				    }

				    //hid_t tmp = H5Pcreate(H5P_DATASET_CREATE);
				    hasChunk = H5Pcreate(H5P_DATASET_CREATE);
				    H5Pset_chunk(hasChunk, ndims, chunk_arr);

				    return hasChunk;
				}
			}
			else if(strcmp(parameter_name, "meta_block_size") == 0) {
				hsize_t block = (hsize_t) strtoull(node->child->value.text.string, NULL, 10);

				if(H5Pset_meta_block_size(fapl_id, block) < 0) {
					fprintf(stderr, "H5Tuner: H5Pset_meta_block_size(%llu) failed\n",
					        (unsigned long long) block);
				}
			}
			else if(strcmp(parameter_name, "chunk_cache") == 0) {
				/*
				 * Only the raw data chunk cache size is tuned.  The slot count and
				 * the preemption policy keep whatever the property list already
				 * holds, so the XML element carries one number rather than four.
				 */
				int mdc_nelmts = 0;
				size_t nslots = 0;
				size_t nbytes = 0;
				double w0 = 0.0;
				size_t wanted = (size_t) strtoull(node->child->value.text.string, NULL, 10);

				if(H5Pget_cache(fapl_id, &mdc_nelmts, &nslots, &nbytes, &w0) < 0) {
					fprintf(stderr, "H5Tuner: H5Pget_cache() failed; chunk_cache not set\n");
				}
				else if(H5Pset_cache(fapl_id, mdc_nelmts, nslots, wanted, w0) < 0) {
					fprintf(stderr, "H5Tuner: H5Pset_cache(rdcc_nbytes=%llu) failed\n",
					        (unsigned long long) wanted);
				}
			}
			else if(strcmp(parameter_name, "coll_metadata_write") == 0) {
#if H5_VERSION_GE(1,10,0)
				if(H5Pset_coll_metadata_write(fapl_id,
				        at_parse_bool(node->child->value.text.string)) < 0) {
					fprintf(stderr, "H5Tuner: H5Pset_coll_metadata_write() failed\n");
				}
#else
				fprintf(stderr, "H5Tuner: coll_metadata_write needs HDF5 1.10.0 or later; "
				        "this build is %d.%d.%d, ignoring\n",
				        H5_VERS_MAJOR, H5_VERS_MINOR, H5_VERS_RELEASE);
#endif
			}
			else if(strcmp(parameter_name, "col_meta_ops") == 0) {
#if H5_VERSION_GE(1,10,0)
				if(H5Pset_all_coll_metadata_ops(fapl_id,
				        at_parse_bool(node->child->value.text.string)) < 0) {
					fprintf(stderr, "H5Tuner: H5Pset_all_coll_metadata_ops() failed\n");
				}
#else
				fprintf(stderr, "H5Tuner: col_meta_ops needs HDF5 1.10.0 or later; "
				        "this build is %d.%d.%d, ignoring\n",
				        H5_VERS_MAJOR, H5_VERS_MINOR, H5_VERS_RELEASE);
#endif
			}
			else if(strcmp(parameter_name, "mdc_conf") == 0) {
				/*
				 * H5AC_cache_config_t has twenty-odd fields and cross-field
				 * constraints (min <= initial <= max, min_clean_fraction in range).
				 * Read the current configuration, override only the sizes the preset
				 * names, and write it back, so every field we do not touch keeps a
				 * value HDF5 already accepted.
				 */
				H5AC_cache_config_t mdc;
				const char *preset = node->child->value.text.string;

				mdc.version = H5AC__CURR_CACHE_CONFIG_VERSION;
				if(H5Pget_mdc_config(fapl_id, &mdc) < 0) {
					fprintf(stderr, "H5Tuner: H5Pget_mdc_config() failed; mdc_conf not set\n");
				}
				else if(strcmp(preset, "default") == 0) {
					/* Leave the library's own metadata cache configuration alone. */
				}
				else if(strcmp(preset, "aggressive") == 0 || strcmp(preset, "conservative") == 0) {
					if(strcmp(preset, "aggressive") == 0) {
						mdc.min_size     = (size_t)   4 * 1024 * 1024;
						mdc.initial_size = (size_t)  32 * 1024 * 1024;
						mdc.max_size     = (size_t) 128 * 1024 * 1024;
					}
					else {
						mdc.min_size     = (size_t) 512 * 1024;
						mdc.initial_size = (size_t)   1 * 1024 * 1024;
						mdc.max_size     = (size_t)   4 * 1024 * 1024;
					}
					mdc.set_initial_size = 1;

					if(H5Pset_mdc_config(fapl_id, &mdc) < 0) {
						fprintf(stderr, "H5Tuner: H5Pset_mdc_config(%s) failed\n", preset);
					}
				}
				else {
					fprintf(stderr, "H5Tuner: unknown mdc_conf preset \'%s\'; expected "
					        "default, aggressive or conservative\n", preset);
				}
			}
		}
		else {
			continue;
		}
	}
	return hasChunk;
}

FORWARD_DECL(H5Fcreate, hid_t, (const char *filename, unsigned flags, hid_t fcpl_id, hid_t fapl_id));
FORWARD_DECL(H5Dwrite, herr_t, (hid_t dataset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t xfer_plist_id, const void * buf));
FORWARD_DECL(H5Dcreate1, hid_t, (hid_t loc_id, const char *name, hid_t type_id, hid_t space_id, hid_t dcpl_id));
FORWARD_DECL(H5Dcreate2, hid_t, (hid_t loc_id, const char *name, hid_t dtype_id, hid_t space_id, hid_t lcpl_id, hid_t dcpl_id, hid_t dapl_id));
#if H5_VERSION_GE(1,12,0)
FORWARD_DECL(H5Fcreate_async, hid_t, (const char *app_file, const char *app_func, unsigned app_line, const char *filename, unsigned flags, hid_t fcpl_id, hid_t fapl_id, hid_t es_id));
FORWARD_DECL(H5Dcreate_async, hid_t, (const char *app_file, const char *app_func, unsigned app_line, hid_t loc_id, const char *name, hid_t type_id, hid_t space_id, hid_t lcpl_id, hid_t dcpl_id, hid_t dapl_id, hid_t es_id));
FORWARD_DECL(H5Dwrite_async, herr_t, (const char *app_file, const char *app_func, unsigned app_line, hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t dxpl_id, const void *buf, hid_t es_id));
#endif

/*
 * Apply every tuned parameter to `fapl_id` and return the filename to open.
 *
 * Split out of the H5Fcreate hook so the async entry point can share it.  That
 * is not a hypothetical need: h5bench calls H5Fcreate_async, not H5Fcreate, and
 * with only the synchronous hook in place the shim loads, runs, and injects
 * nothing -- silently, because a hook that is never called reports nothing.
 * Any application built against HDF5 1.12+ may use the async API this way.
 *
 * `lockless_path` is scratch space the caller owns; the returned pointer is
 * either it or `filename`.
 */
static const char *at_apply_file_params(const char *context,
                                        const char *filename, hid_t fapl_id,
                                        char *lockless_path,
                                        size_t lockless_len)
{
	herr_t ret = -1;
	mxml_node_t *tree = NULL;
	MPI_Comm orig_comm;
	MPI_Info orig_info = MPI_INFO_NULL;
	int parallel = 0;
	const char *effective_filename = filename;

	tree = at_load_config(context);
	if(tree == NULL) {
		return filename;
	}

	/*
	 * Driver check.  This used to print a message and return -1 without ever
	 * calling through, so a serial HDF5 application with LD_PRELOAD set did not
	 * merely go untuned -- it failed to create the file at all.  It is also the
	 * failure mode a sliced I/O kernel walks into if the slicer drops the
	 * H5Pset_fapl_mpio call.
	 *
	 * The HDF5-layer parameters are property list settings that do not involve
	 * MPI, so they are applied either way now.  Only the MPI-IO hints, the file
	 * system hints and the collective metadata switches need the MPI-IO driver.
	 */
	parallel = (H5Pget_driver(fapl_id) == H5FD_MPIO);
	if(parallel) {
		ret = H5Pget_fapl_mpio(fapl_id, &orig_comm, &orig_info);
		if(ret < 0) {
			fprintf(stderr, "H5Tuner: H5Pget_fapl_mpio() failed; applying HDF5 "
			        "parameters only\n");
			parallel = 0;
		}
		else if(orig_info == MPI_INFO_NULL) {
			MPI_Info_create(&orig_info);
		}
	}
	else {
		fprintf(stderr, "H5Tuner: driver is not H5FD_MPIO; skipping MPI-IO and "
		        "file system parameters for %s\n", filename);
	}

	if(parallel) {
		if(gpfs_lockless_filename(tree, filename, lockless_path,
		                          lockless_len)) {
			effective_filename = lockless_path;
		}

		set_mpi_parameter(tree, "IBM_largeblock_io", filename, &orig_info);

		set_mpi_parameter(tree, "striping_factor", filename, &orig_info);
		set_mpi_parameter(tree, "striping_unit", filename, &orig_info);

		set_mpi_parameter(tree, "cb_buffer_size", filename, &orig_info);
		set_mpi_parameter(tree, "cb_nodes", filename, &orig_info);
		set_mpi_parameter(tree, "bgl_nodes_pset", filename, &orig_info);
	}

	set_hdf5_parameter(tree, "sieve_buf_size", filename, fapl_id);
	set_hdf5_parameter(tree, "alignment", filename, fapl_id);
	set_hdf5_parameter(tree, "meta_block_size", filename, fapl_id);
	set_hdf5_parameter(tree, "chunk_cache", filename, fapl_id);
	set_hdf5_parameter(tree, "mdc_conf", filename, fapl_id);

	if(parallel) {
		/* Collective metadata means nothing without the MPI-IO driver; setting it
		 * elsewhere would only produce an error to report. */
		set_hdf5_parameter(tree, "coll_metadata_write", filename, fapl_id);
		set_hdf5_parameter(tree, "col_meta_ops", filename, fapl_id);

		if(H5Pset_fapl_mpio(fapl_id, orig_comm, orig_info) < 0) {
			fprintf(stderr, "H5Tuner: H5Pset_fapl_mpio() failed; MPI-IO hints were "
			        "not applied to %s\n", filename);
		}

		/* H5Pget_fapl_mpio hands back a duplicate that the caller owns, and
		 * H5Pset_fapl_mpio makes its own copy.  Neither was ever freed, so every
		 * H5Fcreate leaked one MPI_Info object. */
		MPI_Info_free(&orig_info);
	}

	/* The parsed document leaked on every call too.  One file is harmless; an
	 * application that creates many files leaked one document each. */
	mxmlDelete(tree);

	return effective_filename;
}

hid_t DECL(H5Fcreate)(const char *filename, unsigned flags, hid_t fcpl_id,
                      hid_t fapl_id)
{
	char lockless_path[1024];
	const char *effective;

	MAP_OR_FAIL(H5Fcreate);

	effective = at_apply_file_params("H5Fcreate", filename, fapl_id,
	                                 lockless_path, sizeof(lockless_path));
	return __fake_H5Fcreate(effective, flags, fcpl_id, fapl_id);
}

#if H5_VERSION_GE(1,12,0)
/*
 * The async variants carry three extra leading arguments that HDF5's own
 * macros fill in (__FILE__, __func__, __LINE__).  They are passed straight
 * through; only the file access property list matters to us.
 */
hid_t DECL(H5Fcreate_async)(const char *app_file, const char *app_func,
                            unsigned app_line, const char *filename,
                            unsigned flags, hid_t fcpl_id, hid_t fapl_id,
                            hid_t es_id)
{
	char lockless_path[1024];
	const char *effective;

	MAP_OR_FAIL(H5Fcreate_async);

	effective = at_apply_file_params("H5Fcreate_async", filename, fapl_id,
	                                 lockless_path, sizeof(lockless_path));
	return __fake_H5Fcreate_async(app_file, app_func, app_line, effective,
	                              flags, fcpl_id, fapl_id, es_id);
}
#endif

herr_t DECL(H5Dwrite)(hid_t dataset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t xfer_plist_id, const void * buf) {
	herr_t ret = -1;

	MAP_OR_FAIL(H5Dwrite);

	#ifdef DEBUG
	  //printf("dataset_id: %d\n", dataset_id);
	  //printf("mem_type_id: %d\n", mem_type_id);
	  //printf("mem_space_id: %d\n", mem_space_id);
  	//printf("file_space_id: %d\n", file_space_id);
  	//printf("xfer_plist_id: %d\n", xfer_plist_id);
	#endif

	#ifdef DEBUG
	  //printf("\nH5Tuner: calling H5Dwrite.\n");
	#endif
	ret = __fake_H5Dwrite(dataset_id, mem_type_id, mem_space_id, file_space_id, xfer_plist_id, buf);

	return ret;
}

hid_t DECL(H5Dcreate1)(hid_t loc_id, const char *name, hid_t type_id, hid_t space_id, hid_t dcpl_id) {
    hid_t ret_value = -1;
    mxml_node_t *tree;

    MAP_OR_FAIL(H5Dcreate1);

    hid_t chunked_pid = -1;

    tree = at_load_config("H5Dcreate1");
    if(tree != NULL) {
        chunked_pid = set_hdf5_parameter(tree, "chunk", name, space_id);
        mxmlDelete(tree);
    }

    #ifdef DEBUG
  	  //printf("\nH5Tuner: calling H5Dcreate1.\n");
  	#endif

    if (chunked_pid == -1) {
		    ret_value = __fake_H5Dcreate1(loc_id, name, type_id, space_id, dcpl_id);
	  }
	  else if(dcpl_id == 0) {
		    ret_value = __fake_H5Dcreate1(loc_id, name, type_id, space_id, chunked_pid);
    }
    else {
		    printf("H5Tuner: Cannot set chunked property list since dcpl_id is not 0!\n");
		    ret_value = __fake_H5Dcreate1(loc_id, name, type_id, space_id, dcpl_id);
    }

    return ret_value;
}

/*
 * Returns a chunked dataset creation property list for `name`, or -1 when the
 * config asks for no chunking.  Shared by the sync and async hooks.
 */
static hid_t at_chunked_dcpl(const char *context, const char *name,
                             hid_t space_id)
{
    hid_t chunked_pid = -1;
    mxml_node_t *tree = at_load_config(context);

    if(tree != NULL) {
        chunked_pid = set_hdf5_parameter(tree, "chunk", name, space_id);
        mxmlDelete(tree);
    }
    return chunked_pid;
}

hid_t DECL(H5Dcreate2)(hid_t loc_id, const char *name, hid_t dtype_id, hid_t space_id, hid_t lcpl_id, hid_t dcpl_id, hid_t dapl_id) {
    hid_t ret_value = -1;

    MAP_OR_FAIL(H5Dcreate2);

    hid_t chunked_pid = at_chunked_dcpl("H5Dcreate2", name, space_id);

    #ifdef DEBUG
  	  //printf("\nH5Tuner: calling H5Dcreate2.\n");
  	#endif

    if (chunked_pid == -1) {
	     ret_value = __fake_H5Dcreate2(loc_id, name, dtype_id, space_id, lcpl_id, dcpl_id, dapl_id);
    }
    else if(dcpl_id == 0) {
	     ret_value = __fake_H5Dcreate2(loc_id, name, dtype_id, space_id, lcpl_id, dcpl_id, dapl_id);
    }
    else {
      printf("H5Tuner: Cannot set chunked property list since dcpl_id is not 0.\n");
       ret_value = __fake_H5Dcreate2(loc_id, name, dtype_id, space_id, lcpl_id, chunked_pid, dapl_id);
    }

    return ret_value;

}

#if H5_VERSION_GE(1,12,0)
hid_t DECL(H5Dcreate_async)(const char *app_file, const char *app_func,
                            unsigned app_line, hid_t loc_id, const char *name,
                            hid_t type_id, hid_t space_id, hid_t lcpl_id,
                            hid_t dcpl_id, hid_t dapl_id, hid_t es_id)
{
	hid_t chunked_pid;

	MAP_OR_FAIL(H5Dcreate_async);

	chunked_pid = at_chunked_dcpl("H5Dcreate_async", name, space_id);
	if(chunked_pid != -1 && dcpl_id != 0) {
		fprintf(stderr, "H5Tuner: cannot set chunked property list since "
		        "dcpl_id is not 0\n");
		dcpl_id = chunked_pid;
	}
	return __fake_H5Dcreate_async(app_file, app_func, app_line, loc_id, name,
	                              type_id, space_id, lcpl_id, dcpl_id,
	                              dapl_id, es_id);
}

/* Pass-through, like the synchronous H5Dwrite hook.  Present so that the
 * symbol is intercepted consistently and so a future change has one place to
 * go; it injects nothing. */
herr_t DECL(H5Dwrite_async)(const char *app_file, const char *app_func,
                            unsigned app_line, hid_t dset_id,
                            hid_t mem_type_id, hid_t mem_space_id,
                            hid_t file_space_id, hid_t dxpl_id,
                            const void *buf, hid_t es_id)
{
	MAP_OR_FAIL(H5Dwrite_async);

	return __fake_H5Dwrite_async(app_file, app_func, app_line, dset_id,
	                             mem_type_id, mem_space_id, file_space_id,
	                             dxpl_id, buf, es_id);
}
#endif
