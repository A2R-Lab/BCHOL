#pragma once
/*  MATH HELPERS - Check later what's in GLASS and delete   */
#include "./help_functions/chol_InPlace.cuh"
#include "./help_functions/diag_Matrix_set.cuh"
#include "./help_functions/dot_product.cuh"
#include "./help_functions/lowerBackSub.cuh"
#include "./help_functions/scaled_sum.cuh"
#include "./help_functions/set_const.cuh"


/*COPY/CSV/DEBUG*/
#include "./help_functions/copy_mult.cuh"
#include "./help_functions/csv.cuh"
#include "./help_functions/print_debug.cuh"



/*Specifically for BCHOL*/
#include "./help_functions/nested_dissect.cuh"
#include "./help_functions/tree_functs.cuh"