/*
 * Copyright (C) 2016, Matthias Rottmann, Artur Strebel, Simon Heybrock, Simone Bacchio, Bjoern Leder, Issaku Kanamori.
 * 
 * This file is part of the DDalphaAMG solver library.
 * 
 * The DDalphaAMG solver library is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * The DDalphaAMG solver library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * 
 * You should have received a copy of the GNU General Public License
 * along with the DDalphaAMG solver library. If not, see http://www.gnu.org/licenses/.
 * 
 */
 
#include "main.h"


global_struct g;
#ifdef HAVE_HDF5
Hdf5_fileinfo h5info;
#endif
struct common_thread_data *commonthreaddata;
struct Thread *no_threading;

int main( int argc, char **argv ) {
    
#ifdef HAVE_HDF5
  h5info.filename=NULL;
  h5info.file_id=-1; 
  h5info.rootgroup_id=-1; 
  h5info.configgroup_id=-1;
  h5info.eigenmodegroup_id=-1;
  h5info.thiseigenmodegroup_id=-1;
  h5info.isOpen=0;
  h5info.mode=-1;
#endif
  level_struct l;
  config_double hopp = NULL, clov = NULL;
  
  MPI_Init( &argc, &argv );
  
  predefine_rank();
  /*if ( g.my_rank == 0 ) {
    printf("\n\n+----------------------------------------------------------------------+\n");
    printf("| The DDalphaAMG solver library.                                       |\n");
    printf("| Copyright (C) 2016, Matthias Rottmann, Artur Strebel,                |\n");
    printf("|       Simon Heybrock, Simone Bacchio, Bjoern Leder, Issaku Kanamori. |\n");
    printf("|                                                                      |\n");
    printf("| This program comes with ABSOLUTELY NO WARRANTY.                      |\n");
    printf("+----------------------------------------------------------------------+\n\n");
  }*/
  
  method_init( &argc, &argv, &l );
  
  no_threading = (struct Thread *)malloc(sizeof(struct Thread));
  setup_no_threading(no_threading, &l);
  
  MALLOC( hopp, complex_double, 3*l.inner_vector_size );
  if ( g.two_cnfgs ) {
    MALLOC( clov, complex_double, 3*l.inner_vector_size );
    printf0("clover term configuration: %s", g.in_clov ); 

    if(g.in_format == _LIME)
      lime_read_conf( (double*)(clov), g.in_clov, &(g.plaq_clov) );
    else if(g.in_format == _MULTI)
      read_conf_multi( (double*)(clov), g.in, &(g.plaq_hopp), &l );
    else
      read_conf( (double*)(clov), g.in_clov, &(g.plaq_clov), &l );

    printf0("hopping term ");
  }

  if(g.in_format == _LIME)
    lime_read_conf( (double*)(hopp), g.in, &(g.plaq_hopp) );
  else if(g.in_format == _MULTI)
    read_conf_multi( (double*)(hopp), g.in, &(g.plaq_hopp), &l );
  else
    read_conf( (double*)(hopp), g.in, &(g.plaq_hopp), &l );

  if ( !g.two_cnfgs ) {
    g.plaq_clov = g.plaq_clov;
  }
  // store configuration, compute clover term
  dirac_setup( hopp, clov, &l );
  FREE( hopp, complex_double, 3*l.inner_vector_size );
  if ( g.two_cnfgs ) {
    FREE( clov, complex_double, 3*l.inner_vector_size );
  }

  commonthreaddata = (struct common_thread_data *)malloc(sizeof(struct common_thread_data));
  init_common_thread_data(commonthreaddata);




#pragma omp parallel num_threads(g.num_openmp_processes)
  {
    g.if_rademacher=0;

    struct Thread threading;
    setup_threading(&threading, commonthreaddata, &l);
    setup_no_threading(no_threading, &l);

    // setup up initial MG hierarchy
    method_setup( NULL, &l, &threading );
    // iterative phase
    method_update( l.setup_iter, &l, &threading );

    complex_double trace, rtrace;
    struct Thread *threadingx = &threading;

    // 1. get rough trace
    l.h_double.nr_levels = 1;
    hutchinson_diver_double_init( &l, &threading );
	  hutchinson_diver_double_alloc( &l, &threading );
    //l.h_double.rough_trace[0] = 0.0;
    l.h_double.rt = 0.0;
    l.h_double.max_iters = 5;
    l.h_double.min_iters = 5;
    rtrace = hutchinson_driver_double( &l, &threading );
	  hutchinson_diver_double_free( &l, &threading );

    // Plain Hutchinson
    // -------------------------------------------------------
    START_MASTER(threadingx)
    if(g.my_rank==0) printf("Computing trace through plain Hutchinson ...\n");
    END_MASTER(threadingx)
    // number of levels for Plain Hutchinson
    l.h_double.nr_levels = 1;
    hutchinson_diver_double_init( &l, &threading );
	  hutchinson_diver_double_alloc( &l, &threading );
    // get actual trace
    //l.h_double.rough_trace[0] = trace;
    l.h_double.rt = rtrace;
    l.h_double.max_iters = 20;
    l.h_double.min_iters = 5;
    trace = hutchinson_driver_double( &l, &threading );
	  hutchinson_diver_double_free( &l, &threading );
    START_MASTER(threadingx)
    if(g.my_rank==0) printf("... done\n");
    END_MASTER(threadingx)
    // -------------------------------------------------------

    START_MASTER(threadingx)
    if(g.my_rank==0) printf("\n");
    END_MASTER(threadingx)

    // MLMC
    // -------------------------------------------------------
    START_MASTER(threadingx)
    if(g.my_rank==0) printf("Computing trace through plain Hutchinson ...\n");
    END_MASTER(threadingx)
    // number of levels for Plain Hutchinson
    l.h_double.nr_levels = g.num_levels;
    hutchinson_diver_double_init( &l, &threading );
	  hutchinson_diver_double_alloc( &l, &threading );
    // get actual trace
    //l->h_double.rough_trace[0] = trace;
    l.h_double.rt = rtrace;
    l.h_double.max_iters = 20;
    l.h_double.min_iters = 5;
    trace = mlmc_hutchinson_diver_double( &l, &threading );
	  hutchinson_diver_double_free( &l, &threading );
    START_MASTER(threadingx)
    if(g.my_rank==0) printf("... done\n");
    END_MASTER(threadingx)
    // -------------------------------------------------------

    //block_hutchinson_driver_double
  }

  finalize_common_thread_data(commonthreaddata);
  finalize_no_threading(no_threading);
  method_free( &l );
  method_finalize( &l );
  
  MPI_Finalize();
  
  return 0;
}
