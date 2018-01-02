#define _POSIX_C_SOURCE 200112L

#include "utils.h"
#include "com_intel_analytics_bigdl_mkl_MklDnn.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "mkldnn.h"
#ifdef WIN32
#include <malloc.h>
#endif

#define BATCH 32

void *aligned_malloc(size_t size, size_t alignment) {
#ifdef WIN32
    return _aligned_malloc(size, alignment);
#else
    void *p;
    return !posix_memalign(&p, alignment, size) ? p : NULL;
#endif
}

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

mkldnn_status_t
prepare_reorder(mkldnn_primitive_t *user_memory,               /** in */
                const_mkldnn_primitive_desc_t *prim_memory_pd, /** in */
                int dir_is_user_to_prim, /** in: user -> prim or prim -> user */
                mkldnn_primitive_t *prim_memory, mkldnn_primitive_t
                *reorder, /** out: reorder primitive created */
                float *buffer)
{
    const_mkldnn_primitive_desc_t user_memory_pd;
    mkldnn_primitive_get_primitive_desc(*user_memory, &user_memory_pd);

    if (!mkldnn_memory_primitive_desc_equal(user_memory_pd, *prim_memory_pd)) {
    printf("need reorder ");
        CHECK(mkldnn_primitive_create(prim_memory, *prim_memory_pd, NULL,
                                      NULL));
        CHECK(mkldnn_memory_set_data_handle(*prim_memory, buffer));

        mkldnn_primitive_desc_t reorder_pd;
        if (dir_is_user_to_prim) {
            /* reorder primitive descriptor doesn't need engine, because it is
             * already appeared in in- and out- memory primitive descriptors */
             printf("reorder 11 ");
            CHECK(mkldnn_reorder_primitive_desc_create(
                    &reorder_pd, user_memory_pd, *prim_memory_pd));
            mkldnn_primitive_at_t inputs = { *user_memory };
            const_mkldnn_primitive_t outputs[] = { *prim_memory };
            CHECK(mkldnn_primitive_create(reorder, reorder_pd, &inputs,
                                          outputs));
        } else {
            printf("reorder 22 ");
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
    CHECK(mkldnn_engine_create(&engine, mkldnn_cpu, 0));

    int net_src_sizes[4] = { BATCH, 3, 227, 227 };
    int net_dst_sizes[4] = { BATCH, 96, 27, 27 };

    float *net_src = (float *)aligned_malloc(product(net_src_sizes,4)*sizeof(float), 64);
    float *net_dst = (float *)aligned_malloc(product(net_dst_sizes, 4)*sizeof(float), 64);

    init_net_data(net_src, 4, net_src_sizes);
    memset(net_dst, 0, product(net_dst_sizes, 4)*sizeof(float));

    /*----------------------------------------------------------------------*/
    /*----------------- Forward Stream -------------------------------------*/
    /* AlexNet: conv
     * {BATCH, 3, 227, 227} (x) {96, 3, 11, 11} -> {BATCH, 96, 55, 55}
     * strides: {4, 4}
     */
    int *conv_src_sizes = net_src_sizes;
    int conv_weights_sizes[4] = { 96, 3, 11, 11 };
    int conv_bias_sizes[4] = { 96 };
    int conv_dst_sizes[4] = { BATCH, 96, 55, 55 };
    int conv_strides[2] = { 4, 4 };
    int conv_padding[2] = { 0, 0 };

    float *conv_src = net_src;
    float *conv_weights = (float *)aligned_malloc(product(conv_weights_sizes, 4)*sizeof(float), 64);
    float *conv_bias = (float *)aligned_malloc(product(conv_bias_sizes, 1)*sizeof(float), 64);

    init_net_data(conv_weights, 4, conv_weights_sizes);
    init_net_data(conv_bias, 1, conv_bias_sizes);

    /* create memory for user data */
    mkldnn_primitive_t conv_user_src_memory, conv_user_weights_memory, conv_user_bias_memory;
    init_data_memory(4, conv_src_sizes, mkldnn_nchw, mkldnn_f32, engine, conv_src, &conv_user_src_memory);
    init_data_memory(4, conv_weights_sizes, mkldnn_nchw, mkldnn_f32, engine, conv_weights, &conv_user_weights_memory);
    init_data_memory(1, conv_bias_sizes, mkldnn_x, mkldnn_f32, engine, conv_bias, &conv_user_bias_memory);

    /* create data descriptors for convolution w/ no specified format */
    mkldnn_memory_desc_t conv_src_md, conv_weights_md, conv_bias_md,
            conv_dst_md;
    CHECK(mkldnn_memory_desc_init(&conv_src_md, 4, conv_src_sizes, mkldnn_f32,
                                  mkldnn_any));
    CHECK(mkldnn_memory_desc_init(&conv_weights_md, 4, conv_weights_sizes,
                                  mkldnn_f32, mkldnn_any));
    CHECK(mkldnn_memory_desc_init(&conv_bias_md, 1, conv_bias_sizes, mkldnn_f32,
                                  mkldnn_x));
    CHECK(mkldnn_memory_desc_init(&conv_dst_md, 4, conv_dst_sizes, mkldnn_f32,
                                  mkldnn_any));

    /* create a convolution */
    mkldnn_convolution_desc_t conv_any_desc;
    CHECK(mkldnn_convolution_forward_desc_init(
            &conv_any_desc, mkldnn_forward, mkldnn_convolution_direct,
            &conv_src_md, &conv_weights_md, &conv_bias_md, &conv_dst_md,
            conv_strides, conv_padding, conv_padding, mkldnn_padding_zero));

    mkldnn_primitive_desc_t conv_pd;
    CHECK(mkldnn_primitive_desc_create(&conv_pd, &conv_any_desc, engine, NULL));

    float *conv_dst_buffer = (float *)aligned_malloc(product(conv_dst_sizes, 4)*sizeof(float), 64);
    memset(conv_dst_buffer, 0, product(conv_dst_sizes, 4)*sizeof(float));
    mkldnn_primitive_t conv_src_memory = conv_user_src_memory;
    mkldnn_primitive_t conv_weights_memory = conv_user_weights_memory;

    /* create memory for dst data, we don't need to reorder it to user data */
    mkldnn_primitive_t conv_internal_dst_memory;
    CHECK(mkldnn_primitive_create(
            &conv_internal_dst_memory,
            mkldnn_primitive_desc_query_pd(conv_pd, mkldnn_query_dst_pd, 0),
            NULL, NULL));
    CHECK(mkldnn_memory_set_data_handle(conv_internal_dst_memory,
                                        conv_dst_buffer));

    mkldnn_primitive_at_t conv_srcs[]
            = { mkldnn_primitive_at(conv_src_memory, 0),
                mkldnn_primitive_at(conv_weights_memory, 0),
                mkldnn_primitive_at(conv_user_bias_memory, 0) };

    const_mkldnn_primitive_t conv_dsts[] = { conv_internal_dst_memory };

    /* finally create a convolution primitive */
    mkldnn_primitive_t conv;
    CHECK(mkldnn_primitive_create(&conv, conv_pd, conv_srcs, conv_dsts));

     uint32_t n_fwd = 0;
    mkldnn_primitive_t net_fwd[10];
    net_fwd[n_fwd++] = conv;

    mkldnn_stream_t stream_fwd;

    int n_iter = 1; //number of iterations for training.
    /* Execute the net */
    while (n_iter) {
        /* Forward pass */
        CHECK(mkldnn_stream_create(&stream_fwd, mkldnn_eager));
        CHECK(mkldnn_stream_submit(stream_fwd, n_fwd, net_fwd, NULL));
        CHECK(mkldnn_stream_wait(stream_fwd, n_fwd, NULL));
        --n_iter;
    }

    /* Cleanup forward */
    CHECK(mkldnn_primitive_desc_destroy(conv_pd));
    mkldnn_stream_destroy(stream_fwd);

    _free(net_src);
    _free(net_dst);

    mkldnn_engine_destroy(engine);

    return mkldnn_success;
}

#include <sys/timeb.h>
#include <unistd.h>

long long getSystemTime() {
    struct timeb t;
    ftime(&t);
    return 1000 * t.time + t.millitm;
}

void test_time()
{
    long long start=getSystemTime();
    int n_iter = 99; //number of iterations for training.
    /* Execute the net */
    while (n_iter) {
        simple_net();
         --n_iter;
    }
    mkldnn_status_t result = simple_net();
    long long end=getSystemTime();
    printf("time: %lld ms\n", end-start);
    return;
}

#ifdef __cplusplus
extern "C" {
#endif
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_All(
  JNIEnv *env, jclass cls) {
    test_time();
}

#ifdef __cplusplus
}
#endif
