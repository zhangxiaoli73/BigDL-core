#include "utils.h"
#include "com_intel_analytics_bigdl_mkl_MklDnn.h"

/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

// Required for posix_memalign
#define _POSIX_C_SOURCE 200112L

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mkldnn.h"
#ifdef WIN32
#include <malloc.h>
#endif

#define BATCH 32

#ifdef WIN32
void _free(void *ptr) {
    _aligned_free(ptr);
}
#else
void _free(void *ptr) {
    free(ptr);
}
#endif

static size_t product(int *arr, size_t size)
{
    size_t prod = 1;
    for (size_t i = 0; i < size; ++i)
        prod *= arr[i];
    return prod;
}

static void init_net_data(float *data, uint32_t dim, const int *dims)
{
    if (dim == 1) {
        for (int i = 0; i < dims[0]; ++i) {
            data[i] = (float)(i % 1637);
        }
    } else if (dim == 4) {
        for (int in = 0; in < dims[0]; ++in) {
            for (int ic = 0; ic < dims[1]; ++ic) {
                for (int ih = 0; ih < dims[2]; ++ih) {
                    for (int iw = 0; iw < dims[3]; ++iw) {
                        int indx = in * dims[1] * dims[2] * dims[3]
                                   + ic * dims[2] * dims[3] + ih * dims[3] + iw;
                        data[indx] = (float)(indx % 1637);
                    }
                }
            }
        }
    }
}

static void init_data_memory(uint32_t dim, const int *dims,
                             mkldnn_memory_format_t user_fmt,
                             mkldnn_data_type_t data_type,
                             mkldnn_engine_t engine, float *data,
                             mkldnn_primitive_t *memory)
{
    mkldnn_memory_desc_t prim_md;
    mkldnn_primitive_desc_t user_pd;
    CHECK(mkldnn_memory_desc_init(&prim_md, dim, dims, data_type, user_fmt));
    CHECK(mkldnn_memory_primitive_desc_create(&user_pd, &prim_md, engine));
    CHECK(mkldnn_primitive_create(memory, user_pd, NULL, NULL));

    void *req = NULL;
    CHECK(mkldnn_memory_get_data_handle(*memory, &req));
    CHECK_TRUE(req == NULL);
    CHECK(mkldnn_memory_set_data_handle(*memory, data));
    CHECK(mkldnn_memory_get_data_handle(*memory, &req));
    CHECK_TRUE(req == data);
    CHECK(mkldnn_primitive_desc_destroy(user_pd));
}

static void init_data_memory_new(uint32_t dim, const int *dims,
                             mkldnn_memory_format_t user_fmt,
                             mkldnn_data_type_t data_type,
                             mkldnn_engine_t engine,
                             mkldnn_primitive_t *memory)
{
    mkldnn_memory_desc_t prim_md;
    mkldnn_primitive_desc_t user_pd;
    CHECK(mkldnn_memory_desc_init(&prim_md, dim, dims, data_type, user_fmt));
    CHECK(mkldnn_memory_primitive_desc_create(&user_pd, &prim_md, engine));
    CHECK(mkldnn_primitive_create(memory, user_pd, NULL, NULL));
    CHECK(mkldnn_primitive_desc_destroy(user_pd));
}

mkldnn_status_t prepare_reorder(mkldnn_primitive_t *user_memory,               /** in */
                const_mkldnn_primitive_desc_t *prim_memory_pd, /** in */
                int dir_is_user_to_prim, /** in: user -> prim or prim -> user */
                mkldnn_primitive_t *prim_memory, mkldnn_primitive_t
                *reorder, /** out: reorder primitive created */
                float *buffer)
{
    const_mkldnn_primitive_desc_t user_memory_pd;
    mkldnn_primitive_get_primitive_desc(*user_memory, &user_memory_pd);

    if (!mkldnn_memory_primitive_desc_equal(user_memory_pd, *prim_memory_pd)) {
        CHECK(mkldnn_primitive_create(prim_memory, *prim_memory_pd, NULL,
                                      NULL));
        CHECK(mkldnn_memory_set_data_handle(*prim_memory, buffer));

        mkldnn_primitive_desc_t reorder_pd;
        if (dir_is_user_to_prim) {
            /* reorder primitive descriptor doesn't need engine, because it is
             * already appeared in in- and out- memory primitive descriptors */
            CHECK(mkldnn_reorder_primitive_desc_create(
                    &reorder_pd, user_memory_pd, *prim_memory_pd));
            mkldnn_primitive_at_t inputs = { *user_memory };
            const_mkldnn_primitive_t outputs[] = { *prim_memory };
            CHECK(mkldnn_primitive_create(reorder, reorder_pd, &inputs,
                                          outputs));
        } else {
            CHECK(mkldnn_reorder_primitive_desc_create(
                    &reorder_pd, *prim_memory_pd, user_memory_pd));
            mkldnn_primitive_at_t inputs = { *prim_memory };
            const_mkldnn_primitive_t outputs[] = { *user_memory };
            CHECK(mkldnn_primitive_create(reorder, reorder_pd, &inputs,
                                          outputs));
        }
        CHECK(mkldnn_primitive_desc_destroy(reorder_pd));
    } else {
        *prim_memory = NULL;
        *reorder = NULL;
    }

    return mkldnn_success;
}

mkldnn_status_t simple_net()
{

    mkldnn_engine_t engine;
    CHECK(mkldnn_engine_create(&engine, mkldnn_cpu, 0 /* idx */));
    /*----------------------------------------------------------------------*/
    /*----------------- Forward Stream -------------------------------------*/
    /* AlexNet: conv
     * {BATCH, 3, 227, 227} (x) {96, 3, 11, 11} -> {BATCH, 96, 55, 55}
     * strides: {4, 4}
     */
    int input_sizes[4] = { 2, 8, 3, 4 };
    int dst_sizes[4] = { 2, 16, 3, 4 };

    /* create data descriptors for convolution w/ no specified format */
    mkldnn_memory_desc_t input1_md, input2_md;
    mkldnn_primitive_desc_t input1_pd, input2_pd;
    CHECK(mkldnn_memory_desc_init(&input1_md, 4, input_sizes, mkldnn_f32, mkldnn_nchw));
    CHECK(mkldnn_memory_desc_init(&input2_md, 4, input_sizes, mkldnn_f32, mkldnn_nchw));

    CHECK(mkldnn_memory_primitive_desc_create(&input1_pd, &input1_md, engine));
    CHECK(mkldnn_memory_primitive_desc_create(&input2_pd, &input2_md, engine));

    mkldnn_memory_desc_t dst_desc;
    mkldnn_memory_desc_init(&dst_desc, 4, dst_sizes, mkldnn_f32, mkldnn_nchw);


    /* create a convolution */
    const_mkldnn_primitive_desc_t input_pds[] = {(const_mkldnn_primitive_desc_t)input1_pd, (const_mkldnn_primitive_desc_t)input2_pd};

    mkldnn_primitive_desc_t concat_desc;
    CHECK(mkldnn_concat_primitive_desc_create(&concat_desc, &dst_desc, 2, 1, input_pds));


    float *input1_buffer = (float *)malloc(product(input_sizes,4)*sizeof(float));
    float *input2_buffer = (float *)malloc(product(input_sizes, 4)*sizeof(float));
    float *dst_buffer = (float *)malloc(product(dst_sizes, 4)*sizeof(float));

    /* create memory for user data */
    mkldnn_primitive_t input1_memory, input2_memory, dst_memory;
    init_data_memory(4, input_sizes, mkldnn_nchw, mkldnn_f32, (mkldnn_engine_t)engine, input1_buffer, &input1_memory);
    init_data_memory(4, input_sizes, mkldnn_nchw, mkldnn_f32, (mkldnn_engine_t)engine, input2_buffer, &input2_memory);
    init_data_memory(4, dst_sizes, mkldnn_nchw, mkldnn_f32, (mkldnn_engine_t)engine, dst_buffer, &dst_memory);

    mkldnn_primitive_at_t concat_srcs[]
            = { mkldnn_primitive_at(input1_memory, 0),
                mkldnn_primitive_at(input2_memory, 0)};

    const_mkldnn_primitive_t concat_dsts[] = { dst_memory };

    /* finally create a convolution primitive */
    mkldnn_primitive_t concat;
    CHECK(mkldnn_primitive_create(&concat, concat_desc, concat_srcs, concat_dsts));

    /* build a simple net */
    uint32_t n_fwd = 0;
    mkldnn_primitive_t net_fwd[10];
    net_fwd[n_fwd++] = concat;

    mkldnn_stream_t stream_fwd;
    CHECK(mkldnn_stream_create(&stream_fwd, mkldnn_eager));
    CHECK(mkldnn_stream_submit(stream_fwd, n_fwd, net_fwd, NULL));
    CHECK(mkldnn_stream_wait(stream_fwd, n_fwd, NULL));

    return mkldnn_success;
}


#ifdef __cplusplus
extern "C" {
#endif

//mkldnn_status_t MKLDNN_API mkldnn_concat_primitive_desc_create(
//        mkldnn_primitive_desc_t *concat_primitive_desc,
//        const mkldnn_memory_desc_t *output_desc, int n, int concat_dimension,
//        const_mkldnn_primitive_desc_t *input_pds);

JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_ConcatExample(
  JNIEnv *env, jclass cls)
{
  simple_net();
}


JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_PrimitiveCreateNew(
  JNIEnv *env, jclass cls,
  long primitive_desc,
  long input1_memory,
  long input2_memory,
  long dst_memory)
{
  mkldnn_primitive_t primitive;
  mkldnn_primitive_at_t concat_srcs[]
      = { mkldnn_primitive_at((const_mkldnn_primitive_t)input1_memory, 0),
          mkldnn_primitive_at((const_mkldnn_primitive_t)input2_memory, 0)};

  const_mkldnn_primitive_t concat_dsts[] = { (const_mkldnn_primitive_t)dst_memory };
  CHECK(
    mkldnn_primitive_create(
      &primitive,
      (const_mkldnn_primitive_desc_t)primitive_desc,
      concat_srcs,
      concat_dsts));

  return (long)primitive;
}

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_ConcatPrimitive(
  JNIEnv *env, jclass cls,
  long output_desc,
  int n,
  int concat_dimension,
  jlongArray input_pds,
  long engine,
  long input11_memory,
  long input21_memory,
  long dst1_memory)
{
  mkldnn_primitive_desc_t concat_desc = malloc(sizeof(mkldnn_primitive_desc_t));

  jlong * j_inputs = (*env)->GetPrimitiveArrayCritical(env, input_pds, JNI_FALSE);
  const_mkldnn_primitive_desc_t srcs[n];
  for (int i = 0; i < n; i++) {
    srcs[i] = (const_mkldnn_primitive_desc_t)(j_inputs[i]);
  }

   CHECK(mkldnn_concat_primitive_desc_create(
     &concat_desc,
     (mkldnn_memory_desc_t *)output_desc,
     n,
     concat_dimension,
     srcs)
  );

  (*env)->ReleasePrimitiveArrayCritical(env, input_pds, j_inputs, 0);

//  int input_sizes[4] = {2, 8, 3, 4};
//  int dst_sizes[4] = {2, 16, 3, 4};

//  mkldnn_primitive_t input1_memory, input2_memory, dst_memory;
//  init_data_memory_new(4, input_sizes, mkldnn_nchw, mkldnn_f32, (mkldnn_engine_t)engine, &input1_memory);
//  init_data_memory_new(4, input_sizes, mkldnn_nchw, mkldnn_f32, (mkldnn_engine_t)engine, &input2_memory);
//  init_data_memory_new(4, dst_sizes, mkldnn_nchw, mkldnn_f32, (mkldnn_engine_t)engine, &dst_memory);

    int input_sizes[4] = { 2, 8, 3, 4 };
    int dst_sizes[4] = { 2, 16, 3, 4 };

    float *input1_buffer = (float *)malloc(product(input_sizes,4)*sizeof(float));
    float *input2_buffer = (float *)malloc(product(input_sizes, 4)*sizeof(float));
    float *dst_buffer = (float *)malloc(product(dst_sizes, 4)*sizeof(float));

    /* create memory for user data */
    mkldnn_primitive_t input1_memory, input2_memory, dst_memory;
    init_data_memory(4, input_sizes, mkldnn_nchw, mkldnn_f32, (mkldnn_engine_t)engine, input1_buffer, &input1_memory);
    init_data_memory(4, input_sizes, mkldnn_nchw, mkldnn_f32, (mkldnn_engine_t)engine, input2_buffer, &input2_memory);
    init_data_memory(4, dst_sizes, mkldnn_nchw, mkldnn_f32, (mkldnn_engine_t)engine, dst_buffer, &dst_memory);

    mkldnn_primitive_at_t concat_srcs[]
            = { mkldnn_primitive_at(input1_memory, 0),
                mkldnn_primitive_at(input2_memory, 0)};

    const_mkldnn_primitive_t concat_dsts[] = { dst_memory };

    /* finally create a convolution primitive */
    mkldnn_primitive_t concat;
    CHECK(mkldnn_primitive_create(&concat, concat_desc, concat_srcs, concat_dsts));

    /* build a simple net */
    uint32_t n_fwd = 0;
    mkldnn_primitive_t net_fwd[10];
    net_fwd[n_fwd++] = concat;

    mkldnn_stream_t stream_fwd;
    CHECK(mkldnn_stream_create(&stream_fwd, mkldnn_eager));
    CHECK(mkldnn_stream_submit(stream_fwd, n_fwd, net_fwd, NULL));
    CHECK(mkldnn_stream_wait(stream_fwd, n_fwd, NULL));

  return (long)concat_desc;
}

#ifdef __cplusplus
}
#endif
