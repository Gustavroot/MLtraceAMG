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

  int i,j, nr_ests=10000;
  int nr_rough_ests=10;
  double trace_tol = 1e-3;
  complex_double trace=0.0;
  complex_double rough_trace=0.0;
  complex_double aux=0.0;
  complex_double estimate[nr_ests]; 
  double variance = 0.0;
  double RMSD = 0.0;

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
  gmres_double_struct* p = &(g.p); //g accesesible from any func.  
  for( i=0; i<nr_ests; i++ ) {
    g.if_rademacher = 1; //Compute random vector
    rhs_define( source, l, threading );
    g.if_rademacher = 0;

    solve( solution, source, l, threading ); //get A⁻¹x

    
    estimate[i] = global_inner_product_double( source, solution, p->v_start, p->v_end, l, threading ); //compute x'A⁻¹x

    aux += estimate[i]; 	 trace = aux/ (i+1); //average the estimates

    for (j=0; j<i; j++) // compute the variance
        variance += (  estimate[j]- trace)*conj(  estimate[j]- trace); 
    variance /= (j+1);

    RMSD = sqrt(variance/(j+1)); //RMSD= sqrt(var+ bias²)

    START_MASTER(threading)
	    //printf( "%d \t var %f \t RMSD %f \t %f + i%f \n ", i, variance, RMSD, CSPLIT(trace)  );
    END_MASTER(threading)
    variance=0.0;
    if(i !=0 && RMSD< cabs(rough_trace)*trace_tol) break;
  }
  

  START_MASTER(threading)
  printf( "trace result = %f+i%f  for %d estimates \n ",CSPLIT(trace), i );
  END_MASTER(threading)

  START_LOCKED_MASTER(threading)
  FREE( solution, complex_double, l->inner_vector_size );
  FREE( source, complex_double, l->inner_vector_size );
  END_LOCKED_MASTER(threading)
}



void block_hutchinson_driver( level_struct *l, struct Thread *threading ) {
  
  

  vector_double solution = NULL, source = NULL, buffer =NULL;
  vector_double* X = NULL;
  int block_size=12, nr_ests=10000;
  complex_double aux[block_size];
  complex_double estimate[block_size*block_size] ;
	int j,i, k, rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  START_MASTER(threading)
  MALLOC( solution, complex_double, l->inner_vector_size );
  MALLOC( source, complex_double, l->inner_vector_size );
  MALLOC( buffer, complex_double, l->inner_vector_size );
  //---------------------------------BIG X------------------------------------------------
  MALLOC( X, vector_double, block_size );
  MALLOC( X[0], complex_double, block_size* l->inner_vector_size);
  // use threading->workspace to distribute pointer to newly allocated memory to all threads
  ((vector_double *)threading->workspace)[0] = solution;
  ((vector_double *)threading->workspace)[1] = source;
  ((vector_double *)threading->workspace)[2] = buffer;
  ((vector_double **)threading->workspace)[3] = X;
  ((vector_double *)threading->workspace)[4] = X[0];
  END_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading)
  
  solution = ((vector_double *)threading->workspace)[0];
  source   = ((vector_double *)threading->workspace)[1];
  buffer   = ((vector_double *)threading->workspace)[2];
  //---------------------------------BIG X------------------------------------------------
  X   = ((vector_double **)threading->workspace)[3];
	X[0] = ((vector_double *)threading->workspace)[4];


  //---------------------------------SET TO ZERO------------------------------------------------
for (i=0; i< block_size*block_size; i++){
  	estimate[i] =0.0;
 }





//-------------------------BLOCK hUTCHINSON ---------  
 // for( k=0; k<nr_ests; k++ ) {

  
  START_MASTER(threading)
   //vector_double_define_random_rademacher( rhs, 0, l->inner_vector_size, l );
   vector_double_define_random_rademacher( buffer, 0, l->inner_vector_size, l );



	
//----------- Setting the pointers to each column of X
	for(i=0; i<block_size; i++)	X[i] = X[0]+i*l->inner_vector_size;

//----------- Initializing Big X
	for(j=0; j<block_size; j++){
  //printf0( "j=%d, iMIN=%d, iMAX %d \n ",   j, j, l->inner_vector_size	);
		for(i=j; i<l->inner_vector_size; i+=block_size){
  		//START_MASTER(threading)
			  //printf( "i = %d j= %d, RANK %d \n ", i, j, rank); 	
 			X[j][i] = buffer[i/12]; 	
  //printf0( "j=%d, i=%d, value %f \n ",   j, i, X[j][i]	);
  		//END_MASTER(threading)
		}
   }
   
   
   
   	for(j=0; j<block_size; j++){
  //printf0( "j=%d, iMIN=%d, iMAX %d \n ",   j, j, l->inner_vector_size	);
		for(i=j; i<l->inner_vector_size; i+=block_size){
  		//START_MASTER(threading)
			  //printf( "i = %d j= %d, RANK %d \n ", i, j, rank); 	
 			X[j][i] = buffer[i/12]; 	
  //printf0( "j=%d, i=%d, value %f \n ",   j, i, X[j][i]	);
  		//END_MASTER(threading)
		}
   }
   
   
   
   
   
    char* a[100] ; 
    
    sprintf(a, "%s%d%s", "BIG",rank, ".txt");
    char fileSpec[strlen(a)+1];
    snprintf( fileSpec, sizeof( fileSpec ), "%s", a );

    
   FILE * fp;
   
   fp = fopen (fileSpec, "w+");
   
   
   for(j=0; j<block_size; j++){
		for(i=0; i<l->inner_vector_size; i++){
		   fprintf(fp, "%f\n", j,i, creal(X[j][i]));
   //fprintf(fp, "%d \t %d\t %f ", j,i, creal(X[j][i]));
   }
      //fprintf(fp, "\n");
   }
   fclose(fp);
   
  END_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading)

/*
//SOLVING PART
  solve( solution, X[0], l, threading ); //get A⁻¹x
  
  gmres_double_struct* p = &(g.p);
 // for (i=0; i< block_size; i++){
  	aux[k] = global_inner_product_double( X[0], solution, p->v_start, p->v_end, l, threading ); //compute x'A⁻¹x
  	estimate[0] += aux[k];
  	
  	   //printf0( "i = %d estimate %f + i%f  \n ", i, CSPLIT(estimate[0])); 
  //}
*/



//} //LOOP FOR K







//PRINTING ESTIMATES
/*for (i=0; i< block_size; i++){
  	estimate[i] /=nr_ests;
  }



START_MASTER(threading)

  //for(i=0; i<block_size; i++){
  printf( "i = %d estimate %f + i%f  \n ", i, CSPLIT(estimate[0])); 	
//}
  END_MASTER(threading)


*/

  
  
  
  START_LOCKED_MASTER(threading)
  FREE( solution, complex_double, l->inner_vector_size );
  FREE( source, complex_double, l->inner_vector_size );
  FREE( buffer, complex_double, l->inner_vector_size);

  FREE( X[0], complex_double, block_size* l->inner_vector_size);
  FREE( X, vector_double, block_size );

  END_LOCKED_MASTER(threading)
}


/* Testing buffer
  
int i;
START_MASTER(threading)

  for(i=0; i<block_size*block_size; i++){
  printf0( "i = %d buffer %f  \n ", i, creal(buffer[i])); 	
}
  END_MASTER(threading)

*/




