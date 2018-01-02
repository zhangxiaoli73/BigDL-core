#include "utils.h"
#include "com_intel_analytics_bigdl_mkl_MklDnn.h"

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_PrimitiveDescCreate(
  JNIEnv *env, jclass cls,
  long op_desc, long engine,
  long hint_forward_primitive_desc)
{
  mkldnn_primitive_desc_t *primitive_desc =
    malloc(sizeof(mkldnn_primitive_desc_t));

  mkldnn_engine_t *j_engine = (mkldnn_engine_t *)engine;
//  mkldnn_engine_t *j_engine = malloc(sizeof(mkldnn_engine_t));
//
//  CHECK(mkldnn_engine_create(j_engine,
//  (mkldnn_engine_kind_t)mkldnn_cpu,
//  (size_t)0));

  CHECK(
    mkldnn_primitive_desc_create(
      primitive_desc,
      (const_mkldnn_op_desc_t)op_desc,
      *j_engine,
      (const_mkldnn_primitive_desc_t)hint_forward_primitive_desc));

  return (long)primitive_desc;
}

JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_PrimitiveDescDestroy(
  JNIEnv *env, jclass cls,
  long primitive_desc)
{
  mkldnn_primitive_desc_t *j_primitive_desc =
    (mkldnn_primitive_desc_t *)primitive_desc;

  CHECK(
    mkldnn_primitive_desc_destroy(*j_primitive_desc));
  free(j_primitive_desc);
}

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_PrimitiveCreate(
  JNIEnv *env, jclass cls,
  long primitive_desc,
  jlongArray inputs, // TODO java array
  jlongArray outputs) // java array
{
  jlong * j_inputs = (*env)->GetPrimitiveArrayCritical(env, inputs, JNI_FALSE);
  jlong * j_outputs = (*env)->GetPrimitiveArrayCritical(env, outputs, JNI_FALSE);
  mkldnn_primitive_t *primitive = malloc(sizeof(mkldnn_primitive_t));

  CHECK(
    mkldnn_primitive_create(
      primitive,
      *((const_mkldnn_primitive_desc_t *)primitive_desc),
      (mkldnn_primitive_at_t *)j_inputs,
      (const_mkldnn_primitive_t *)j_outputs));

  (*env)->ReleasePrimitiveArrayCritical(env, inputs, j_inputs, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, outputs, j_outputs, 0);

  return (long)primitive;
}


JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_PrimitiveCreateNoPointer(
  JNIEnv *env, jclass cls,
  long primitive_desc)
{
//  jlong * j_inputs = (*env)->GetPrimitiveArrayCritical(env, inputs, JNI_FALSE);
//  jlong * j_outputs = (*env)->GetPrimitiveArrayCritical(env, outputs, JNI_FALSE);
  mkldnn_primitive_t *primitive = malloc(sizeof(mkldnn_primitive_t));

  CHECK(
    mkldnn_primitive_create(
      primitive,
      (const_mkldnn_primitive_desc_t)primitive_desc,
      NULL,
      NULL)
     );

//  (*env)->ReleasePrimitiveArrayCritical(env, inputs, j_inputs, 0);
//  (*env)->ReleasePrimitiveArrayCritical(env, outputs, j_outputs, 0);

  return (long)primitive;
}

JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_PrimitiveDestroy(
  JNIEnv *env, jclass cls,
  long primitive)
{
  mkldnn_primitive_t *j_primitive = (mkldnn_primitive_t *)primitive;
  mkldnn_primitive_destroy(*j_primitive);
  free(j_primitive);
}

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_PrimitiveCreateForSubmit(
  JNIEnv *env, jclass cls,
  long primitive_desc,
  jlongArray inputs, // TODO java array
  int length_inputs,
  jlongArray outputs,
  int length_outputs) // java array
{
  jlong * j_inputs = (*env)->GetPrimitiveArrayCritical(env, inputs, JNI_FALSE);
  jlong * j_outputs = (*env)->GetPrimitiveArrayCritical(env, outputs, JNI_FALSE);
  mkldnn_primitive_t *primitive = malloc(sizeof(mkldnn_primitive_t));

  mkldnn_primitive_at_t primitive_at[length_inputs];
  const_mkldnn_primitive_t const_primitive[length_outputs];
  int i = 0;
  while (i < length_inputs) {
    const_mkldnn_primitive_t *temp = (const_mkldnn_primitive_t *)j_inputs[i];
    primitive_at[i] = mkldnn_primitive_at(*temp, 0);
    i ++;
  }
  i = 0;
  while (i < length_outputs) {
    const_mkldnn_primitive_t *temp = (const_mkldnn_primitive_t *)j_outputs[i];
    const_primitive[i] = *temp;
    i ++;
  }

  CHECK(
    mkldnn_primitive_create(
      primitive,
      *((const_mkldnn_primitive_desc_t *)primitive_desc),
      primitive_at,
      const_primitive)
  );

  (*env)->ReleasePrimitiveArrayCritical(env, inputs, j_inputs, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, outputs, j_outputs, 0);

  return (long)primitive;
}

// output is const_mkldnn_primitive_desc_t

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_ReorderPrimitiveDescCreate(
  JNIEnv *env, jclass cls, long input, long output) {
     mkldnn_primitive_desc_t *reorder_primitive_desc =  malloc(sizeof(mkldnn_primitive_desc_t));

     CHECK(
       mkldnn_reorder_primitive_desc_create(
         reorder_primitive_desc,
         *((const_mkldnn_primitive_desc_t *)input),
         (const_mkldnn_primitive_desc_t)output)
     );

     return (long)reorder_primitive_desc;
  }

// output is const_mkldnn_primitive_desc_t
JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_ReorderPrimitiveDescCreateOuptut(
  JNIEnv *env, jclass cls, long input, long output) {
     mkldnn_primitive_desc_t *reorder_primitive_desc =  malloc(sizeof(mkldnn_primitive_desc_t));

     CHECK(
       mkldnn_reorder_primitive_desc_create(
         reorder_primitive_desc,
         *((const_mkldnn_primitive_desc_t *)input),
         (const_mkldnn_primitive_desc_t)output)
       );

     return (long)reorder_primitive_desc;
  }

// input is  const_mkldnn_primitive_desc_t
JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_ReorderPrimitiveDescCreateInput(
  JNIEnv *env, jclass cls, long input, long output) {

   mkldnn_primitive_desc_t *reorder_primitive_desc =  malloc(sizeof(mkldnn_primitive_desc_t));

   CHECK(
     mkldnn_reorder_primitive_desc_create(
       reorder_primitive_desc,
       (const_mkldnn_primitive_desc_t)input,
       *((const_mkldnn_primitive_desc_t *)output))
   );
   return (long)reorder_primitive_desc;
}

/** Compares two descriptors of memory primitives.
  * @return 1 if the descriptors are the same.
  * @return 0 if the descriptors are different.
  *
  * Use this function to identify whether a reorder is required for the memory
  * primitives.
  */
JNIEXPORT int JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_MemoryPrimitiveDescEqual(
  JNIEnv *env, jclass cls, long lhs, long rhs) {
     return mkldnn_memory_primitive_desc_equal(
         *((const_mkldnn_primitive_desc_t *)lhs),
         *((const_mkldnn_primitive_desc_t *)rhs));
  }

///** Queries primitive descriptor for primitive descriptor
//*
//* @returns NULL in case of any error */
//const_mkldnn_primitive_desc_t MKLDNN_API mkldnn_primitive_desc_query_pd(
//      const_mkldnn_primitive_desc_t primitive_desc, mkldnn_query_t what,
//      int index);
//
//JNIEXPORT int JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_PrimitiveDescQueryPd(
//  JNIEnv *env, jclass cls, long lhs, long rhs) {
//
//     const_mkldnn_primitive_desc_t *query_pd =  malloc(sizeof(const_mkldnn_primitive_desc_t));
//
//
//
//     return mkldnn_memory_primitive_desc_equal(
//         *((const_mkldnn_primitive_desc_t *)input),
//         *((const_mkldnn_primitive_desc_t *)output));
//  }


/** Retrieves a reference to the @p primitive_desc descriptor of given @p
 * primitive.
 *
 * @warning
 *     Returned object must not be destroyed by user. 'const' qualifier of the
 *     returned object prevents such attempts. */
JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_PrimitiveGetPrimitiveDesc(
  JNIEnv *env, jclass cls, long primitive) {

     const_mkldnn_primitive_desc_t *primitive_desc =  malloc(sizeof(const_mkldnn_primitive_desc_t));
     CHECK(mkldnn_primitive_get_primitive_desc(*((const_mkldnn_primitive_t *)primitive), primitive_desc));
     return (long)primitive_desc;
  }

/** Queries primitive descriptor for primitive descriptor
 *
 * @returns NULL in case of any error */
JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_PrimitiveDescQueryPd(
  JNIEnv *env, jclass cls, long primitive, int what, int index) {

    const_mkldnn_primitive_desc_t pd;
    // pd = mkldnn_primitive_desc_query_pd(*((const_mkldnn_primitive_desc_t *)primitive), (mkldnn_query_t)what, index);

    mkldnn_query_t t;
    if (what == 1) {
      t = mkldnn_query_src_pd;
    } else if (what == 2) {
      t = mkldnn_query_weights_pd;
    } else if (what == 3) {
      t = mkldnn_query_dst_pd;
    } else if (what == 4) {
      t = mkldnn_query_diff_dst_pd; //gradOutput
    } else if (what == 5) {
      t = mkldnn_query_diff_src_pd; //gradInput
    } else if (what == 6) {
      t = mkldnn_query_diff_weights_pd; //gradWeight
    } else {
      t = mkldnn_query_workspace_pd;
    }
    pd = mkldnn_primitive_desc_query_pd(*((const_mkldnn_primitive_desc_t *)primitive), t, index);
    return (long)pd;
}

#ifdef __cplusplus
}
#endif
