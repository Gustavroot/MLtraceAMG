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

void rhs_define( vector_double rhs, level_struct *l, struct Thread *threading ) {
  
  // no hyperthreading here
  if(threading->thread != 0)
    return;

  int start = threading->start_index[l->depth];
  int end = threading->end_index[l->depth];

  if ( g.rhs == 0 ) {
    vector_double_define( rhs, 1, start, end, l );
    START_MASTER(threading)
    if ( g.print > 0 ) printf0("rhs = ones\n");
    END_MASTER(threading)
  } else if ( g.rhs == 1 )  {
    vector_double_define( rhs, 0, start, end, l );
    if ( g.my_rank == 0 ) {
      START_LOCKED_MASTER(threading)
      rhs[0] = 1.0;
      END_LOCKED_MASTER(threading)
    }
    START_MASTER(threading)
    if ( g.print > 0 ) printf0("rhs = first unit vector\n");
    END_MASTER(threading)
  } else if ( g.rhs == 2 ) {
    // this would yield different results if we threaded it, so we don't
    START_LOCKED_MASTER(threading)
    if ( g.if_rademacher==1 )
      vector_double_define_random_rademacher( rhs, 0, l->inner_vector_size, l );
    else
      vector_double_define_random( rhs, 0, l->inner_vector_size, l );
    END_LOCKED_MASTER(threading)
    START_MASTER(threading)
    if ( g.print > 0 ) printf0("rhs = random\n");
    END_MASTER(threading)
  //} else if ( g.rhs == 4 ) {
  //  // this would yield different results if we threaded it, so we don't
  //  START_LOCKED_MASTER(threading)
  //  vector_double_define_random_rademacher( rhs, 0, l->inner_vector_size, l );
  //  END_LOCKED_MASTER(threading)
  //  START_MASTER(threading)
  //  if ( g.print > 0 ) printf0("rhs = random\n");
  //  END_MASTER(threading)
  } else if ( g.rhs == 3 ) {
    vector_double_define( rhs, 0, start, end, l );
  } else {
    ASSERT( g.rhs >= 0 && g.rhs <= 4 );
  }
}


int wilson_driver( vector_double solution, vector_double source, level_struct *l, struct Thread *threading ) {
  
  int iter = 0, start = threading->start_index[l->depth], end = threading->end_index[l->depth];
  
  vector_double rhs = g.mixed_precision==2?g.p_MP.dp.b:g.p.b;
  vector_double sol = g.mixed_precision==2?g.p_MP.dp.x:g.p.x;

#ifdef WILSON_BENCHMARK
  START_MASTER(threading)
  prof_init( l );
  END_MASTER(threading)
  double t = -MPI_Wtime();
  double t_min = 1000;
  for ( int i=0; i<100; i++ ) {
    double tmp_t = -MPI_Wtime();
#endif
  
  vector_double_copy( rhs, source, start, end, l );  
  if ( g.method == -1 ) {
    cgn_double( &(g.p), l, threading );
  } else if ( g.mixed_precision == 2 ) {
    iter = fgmres_MP( &(g.p_MP), l, threading );
  } else {
    iter = fgmres_double( &(g.p), l, threading );
  }
  vector_double_copy( solution, sol, start, end, l );
#ifdef WILSON_BENCHMARK
    tmp_t += MPI_Wtime();
    if ( tmp_t < t_min )
      t_min = tmp_t;
  }
  t +=MPI_Wtime();
  START_MASTER(threading)
  printf0("average over 100 solves: %lf seconds\n", t/100 );
  printf0("minimum out of 100 solves: %lf seconds\n", t_min );
  prof_print( l );
  END_MASTER(threading)
#endif
  
  return iter;
}


void solve( vector_double solution, vector_double source, level_struct *l, struct Thread *threading ) {
  
  vector_double rhs = g.mixed_precision==2?g.p_MP.dp.b:g.p.b;

  if ( g.vt.evaluation ) {
    // this would yield different results if we threaded it, so we don't
    START_LOCKED_MASTER(threading)
    vector_double_define_random( rhs, 0, l->inner_vector_size, l );
    scan_var( &(g.vt), l );
    END_LOCKED_MASTER(threading)
  } else {
    wilson_driver( solution, source, l, threading );
  }
}


void solve_driver( level_struct *l, struct Thread *threading ) {
  
  vector_double solution = NULL, source = NULL;
  
  START_MASTER(threading)
  MALLOC( solution, complex_double, l->inner_vector_size );
  MALLOC( source, complex_double, l->inner_vector_size );
  // use threading->workspace to distribute pointer to newly allocated memory to all threads
  ((vector_double *)threading->workspace)[0] = solution;
  ((vector_double *)threading->workspace)[1] = source;
  END_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading)
  solution = ((vector_double *)threading->workspace)[0];
  source   = ((vector_double *)threading->workspace)[1];
  
  rhs_define( source, l, threading );
  
  solve( solution, source, l, threading );

  START_LOCKED_MASTER(threading)
  FREE( solution, complex_double, l->inner_vector_size );
  FREE( source, complex_double, l->inner_vector_size );
  END_LOCKED_MASTER(threading)
}

void hutchinson_driver( level_struct *l, struct Thread *threading ) {
  
  vector_double solution = NULL, source = NULL;

  int i,j, nr_ests=100000;
  int nr_rough_ests=10;
  double trace_tol = 1e-4;
  complex_double trace=0.0;
  complex_double rough_trace=0.0;
  complex_double aux=0.0;
  complex_double estimate[nr_ests]; 
  complex_double variance = 0.0;
  complex_double RMSD = 0.0;

  START_MASTER(threading)
  MALLOC( solution, complex_double, l->inner_vector_size );
  MALLOC( source, complex_double, l->inner_vector_size );
  // use threading->workspace to distribute pointer to newly allocated memory to all threads
  ((vector_double *)threading->workspace)[0] = solution;
  ((vector_double *)threading->workspace)[1] = source;
  END_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading)

  solution = ((vector_double *)threading->workspace)[0];
  source   = ((vector_double *)threading->workspace)[1];

  for( i=0; i<nr_rough_ests; i++ ) {
    g.if_rademacher = 1; //Compute random vector
    rhs_define( source, l, threading );
    g.if_rademacher = 0;

    solve( solution, source, l, threading ); //get A⁻¹x

    gmres_double_struct* p = &(g.p);
    estimate[i] = global_inner_product_double( source, solution, p->v_start, p->v_end, l, threading ); //compute x'A⁻¹x

    aux += estimate[i]; 	 rough_trace = aux/ (i+1); //average the estimates

  }
    START_MASTER(threading)
	    printf( "Rough Trace: %f + i%f \n ", CSPLIT(rough_trace)  );
    END_MASTER(threading)

  aux=0.0; 
  for( i=0; i<nr_ests; i++ ) {
    g.if_rademacher = 1; //Compute random vector
    rhs_define( source, l, threading );
    g.if_rademacher = 0;

    solve( solution, source, l, threading ); //get A⁻¹x

    gmres_double_struct* p = &(g.p);
    estimate[i] = global_inner_product_double( source, solution, p->v_start, p->v_end, l, threading ); //compute x'A⁻¹x

    aux += estimate[i]; 	 trace = aux/ (i+1); //average the estimates

    for (j=0; j<i; j++) // compute the variance
        variance += (  estimate[j]- trace)*conj(  estimate[j]- trace); 
    variance /= (j+1);

    RMSD = sqrt(variance/(j+1)); //RMSD= sqrt(var+ bias²)

    START_MASTER(threading)
	    printf( "%d \t var %f \t RMSD %f \t %f + i%f \n ", i, creal(variance), creal(RMSD), CSPLIT(trace)  );
    END_MASTER(threading)
    variance=0.0;
    if(i !=0 && cabs(RMSD)< cabs(rough_trace)*trace_tol) break;
  }
  

  START_MASTER(threading)
  printf( "trace result = %f+i%f  for %d estimates \n ",CSPLIT(trace), i );
  END_MASTER(threading)

  START_LOCKED_MASTER(threading)
  FREE( solution, complex_double, l->inner_vector_size );
  FREE( source, complex_double, l->inner_vector_size );
  END_LOCKED_MASTER(threading)
}
