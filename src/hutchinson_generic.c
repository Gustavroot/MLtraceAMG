
#include "main.h"



void rhs_define_PRECISION( vector_PRECISION rhs, level_struct *l, struct Thread *threading ) {
  
  // no hyperthreading here
  if(threading->thread != 0)
    return;

  int start = threading->start_index[l->depth];
  int end = threading->end_index[l->depth];

  if ( g.rhs == 0 ) {
    vector_PRECISION_define( rhs, 1, start, end, l );
    START_MASTER(threading)
    if ( g.print > 0 ) printf0("rhs = ones\n");
    END_MASTER(threading)
  } else if ( g.rhs == 1 )  {
    vector_PRECISION_define( rhs, 0, start, end, l );
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
      vector_PRECISION_define_random_rademacher( rhs, 0, l->inner_vector_size, l );
    else
      vector_PRECISION_define_random( rhs, 0, l->inner_vector_size, l );
    END_LOCKED_MASTER(threading)
    START_MASTER(threading)
    if ( g.print > 0 ) printf0("rhs = random\n");
    END_MASTER(threading)
  //} else if ( g.rhs == 4 ) {
  //  // this would yield different results if we threaded it, so we don't
  //  START_LOCKED_MASTER(threading)
  //  vector_PRECISION_define_random_rademacher( rhs, 0, l->inner_vector_size, l );
  //  END_LOCKED_MASTER(threading)
  //  START_MASTER(threading)
  //  if ( g.print > 0 ) printf0("rhs = random\n");
  //  END_MASTER(threading)
  } else if ( g.rhs == 3 ) {
    vector_PRECISION_define( rhs, 0, start, end, l );
  } else {
    ASSERT( g.rhs >= 0 && g.rhs <= 4 );
  }
}

void solve_PRECISION( vector_PRECISION solution, vector_PRECISION source, level_struct *l, struct Thread *threading ){

  int iter = 0, start = threading->start_index[l->depth], end = threading->end_index[l->depth];

  vector_PRECISION rhs = g.mixed_precision==2?g.p_MP.dp.b:g.p.b;
  vector_PRECISION sol = g.mixed_precision==2?g.p_MP.dp.x:g.p.x;

  vector_PRECISION_copy( rhs, source, start, end, l );  
  iter = fgmres_PRECISION( &(g.p), l, threading );
  vector_PRECISION_copy( solution, sol, start, end, l );
}


void hutchinson_driver_PRECISION( level_struct *l, struct Thread *threading ) {
  
  vector_PRECISION solution = NULL, source = NULL;

  int i,j, nr_ests=10000;
  int nr_rough_ests=10;
  PRECISION trace_tol = 1e-4;
  complex_PRECISION trace=0.0;
  complex_PRECISION rough_trace=0.0;
  complex_PRECISION aux=0.0;
  complex_PRECISION sample[nr_ests]; 
  PRECISION variance = 0.0;
  PRECISION RMSD = 0.0;

  START_MASTER(threading)
  MALLOC( solution, complex_PRECISION, l->inner_vector_size );
  MALLOC( source, complex_PRECISION, l->inner_vector_size );
  // use threading->workspace to distribute pointer to newly allocated memory to all threads
  ((vector_PRECISION *)threading->workspace)[0] = solution;
  ((vector_PRECISION *)threading->workspace)[1] = source;
  END_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading)

  solution = ((vector_PRECISION *)threading->workspace)[0];
  source   = ((vector_PRECISION *)threading->workspace)[1];

  for( i=0; i<nr_rough_ests; i++ ) {
    g.if_rademacher = 1; //Compute random vector
    rhs_define_PRECISION( source, l, threading );
    g.if_rademacher = 0;

    solve_PRECISION( solution, source, l, threading ); //get A⁻¹x

    gmres_PRECISION_struct* p = &(g.p);
    sample[i] = global_inner_product_PRECISION( source, solution, p->v_start, p->v_end, l, threading ); //compute x'A⁻¹x

    aux += sample[i]; 	 rough_trace = aux/ (i+1); //average the samples

  }

  START_MASTER(threading)
  printf( "Rough Trace: %f + i%f \n ", CSPLIT(rough_trace)  );
  END_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading)

  aux=0.0; 
  gmres_PRECISION_struct* p = &(g.p); //g accesesible from any func.  
  for( i=0; i<nr_ests; i++ ) {
    g.if_rademacher = 1; //Compute random vector
    rhs_define_PRECISION( source, l, threading );
    g.if_rademacher = 0;

    solve_PRECISION( solution, source, l, threading ); //get A⁻¹x

    sample[i] = global_inner_product_PRECISION( source, solution, p->v_start, p->v_end, l, threading ); //compute x'A⁻¹x

    aux += sample[i]; 	 trace = aux/ (i+1); //average the samples

    for (j=0; j<i; j++) // compute the variance
        variance += (  sample[j]- trace)*conj(  sample[j]- trace); 
    variance /= (j+1);

    RMSD = sqrt(variance/(j+1)); //RMSD= sqrt(var+ bias²)

    START_MASTER(threading)
	    printf( "%d \t var %f \t RMSD %f \t %f + i%f \n ", i, variance, RMSD, CSPLIT(trace)  );
    END_MASTER(threading)
    SYNC_MASTER_TO_ALL(threading)
    variance=0.0;
    if(i !=0 && RMSD< cabs(rough_trace)*trace_tol) break;
  }
  
  

  START_MASTER(threading)
  printf( "trace result = %f+i%f  for %d samples \n ",CSPLIT(trace), i );
  END_MASTER(threading)
	SYNC_MASTER_TO_ALL(threading)
	
  START_LOCKED_MASTER(threading)
  FREE( solution, complex_PRECISION, l->inner_vector_size );
  FREE( source, complex_PRECISION, l->inner_vector_size );
  END_LOCKED_MASTER(threading)
}



void block_hutchinson_PRECISION( level_struct *l, struct Thread *threading , complex_PRECISION* estimate, complex_PRECISION* variance ){
	vector_PRECISION solution = l->h_PRECISION.buffer2; vector_PRECISION* X = l->h_PRECISION.X;
	vector_PRECISION buffer = l->h_PRECISION.buffer1; vector_PRECISION sample = l->h_PRECISION.sample;
	vector_PRECISION rough_trace = l->h_PRECISION.rough_trace; vector_PRECISION total_variance = l->h_PRECISION.total_variance;
	complex_PRECISION trace = l->h_PRECISION.trace;
	gmres_PRECISION_struct* p = &(g.p);
	int k, i, j, v, counter=0;
	int block_size=l->h_PRECISION.block_size, nr_ests=l->h_PRECISION.buffer_int1; 
  PRECISION trace_tol = l->h_PRECISION.trace_tol;
  
	for( k=0; k<nr_ests; k++ ) { //Hitchinson Method loop
		START_MASTER(threading)
	 	// Initialize buffer with Rademacher
		vector_PRECISION_define_random_rademacher( buffer, 0, l->inner_vector_size/block_size, l );		
		//---------------------- Fill Big X
		for(j=0; j<block_size; j++)
			for(i=j; i<l->inner_vector_size; i+=block_size)
	 				X[j][i] = buffer[i/12]; 					
		END_MASTER(threading)
		SYNC_MASTER_TO_ALL(threading)

	//-----------------------COMPUTING THE BLOCK TRACE SAMPLE
		for(j=0; j<block_size; j++){	
	 		solve_PRECISION( solution, X[j], l, threading ); //get A⁻¹x 
			for (i=0; i< block_size; i++){
				complex_PRECISION tmpx = global_inner_product_PRECISION( X[i], solution, p->v_start, p->v_end, l, threading ); //compute x'A⁻¹x   	
				START_MASTER(threading)
				sample[j+i*block_size+ k*block_size*block_size ] = tmpx;
				estimate[j+i*block_size] += sample[j+i*block_size+ k*block_size*block_size ];   
									
				for (v=0; v<k; v++) // compute the variance for the (i,j) element in the BT
        	variance[j+i*block_size] += (  sample[j+i*block_size+ v*block_size*block_size ]- estimate[j+i*block_size]/(k+1)  )*conj(  sample[j+i*block_size+ v*block_size*block_size ]- estimate[j+i*block_size]/(k+1) ); 
        		 				
  		  variance[j+i*block_size] /= (v+1);	
				END_MASTER(threading)			
				SYNC_MASTER_TO_ALL(threading)
	 		}		  		
	 	}	//end of Block sample  	
	 		
	 	//Compute Total Variation	
	 	START_MASTER(threading) 
 		for (v=0; v< block_size*block_size; v++){
			total_variance[0]+= variance[v]; 
	 	}
	 	END_MASTER(threading)
	 	SYNC_MASTER_TO_ALL(threading)
   
		counter=k;
		if(k !=0 && sqrt(total_variance[0])/(k+1) < cabs(rough_trace[0])*trace_tol){ if(k>4) {counter=k; break;} }
	//	if(g.my_rank==0){	
			START_MASTER(threading) 
			printf( "%d \t total_var %f \t RMSD %f \t first entry %f \t rough_trace %f\n ", k, creal(total_variance[0]), sqrt(creal(total_variance[0])/(k+1)), creal(estimate[0])/(k+1), creal(rough_trace[0])  );
			END_MASTER(threading)
   		fflush(stdout);
	//	}	
		total_variance[0]=0.0;  //Reset Total Variation
	} //LOOP FOR K
		
	START_MASTER(threading)

	for (i=0; i< block_size*block_size; i++)	{estimate[i] /=(counter+1); } //Average each estimate		
	for (i=0; i< block_size*block_size; i+=block_size+1){  trace += estimate[i];} //Compute trace
	
	l->h_PRECISION.trace=trace;
  END_MASTER(threading)
	SYNC_MASTER_TO_ALL(threading)
}






void block_hutchinson_driver_PRECISION( level_struct *l, struct Thread *threading ) {

  if(g.my_rank==0){
    START_MASTER(threading)
    printf("Inside block_hutchinson_driver(...) ...\n");
    END_MASTER(threading)
    fflush(stdout);
  }
	l->h_PRECISION.block_size =12; l->h_PRECISION.nr_ests= 1000; l->h_PRECISION.nr_rough_ests =5;  l->h_PRECISION.trace_tol = 1e-3;
  l->h_PRECISION.trace=0.0;
  
  vector_PRECISION solution = l->h_PRECISION.buffer2, source = NULL, buffer =l->h_PRECISION.buffer1, sample=l->h_PRECISION.sample;
  vector_PRECISION* X = l->h_PRECISION.X; vector_PRECISION total_variance= l->h_PRECISION.total_variance; 
  vector_PRECISION rough_trace = l->h_PRECISION.rough_trace;
  complex_PRECISION trace= l->h_PRECISION.trace; 
  
  int block_size=l->h_PRECISION.block_size, nr_ests=l->h_PRECISION.nr_ests; 
  int nr_rough_ests=l->h_PRECISION.nr_rough_ests;  
  int i;
	
  complex_PRECISION estimate[block_size*block_size] ;
  complex_PRECISION variance[block_size*block_size] ;
	

  START_MASTER(threading)
  MALLOC( l->h_PRECISION.buffer2, complex_PRECISION, l->inner_vector_size );
  MALLOC( source, complex_PRECISION, l->inner_vector_size );
  MALLOC( l->h_PRECISION.buffer1, complex_PRECISION, l->inner_vector_size/block_size );
  //---------------------------------BIG X------------------------------------------------
  MALLOC( l->h_PRECISION.X, vector_PRECISION, block_size );
  MALLOC( l->h_PRECISION.X[0], complex_PRECISION, block_size* l->inner_vector_size);
  MALLOC( l->h_PRECISION.sample, complex_PRECISION, block_size*block_size*nr_ests);
  MALLOC( l->h_PRECISION.total_variance, complex_PRECISION, 1);
  MALLOC( l->h_PRECISION.rough_trace, complex_PRECISION, 1);
  // use threading->workspace to distribute pointer to newly allocated memory to all threads
  ((vector_PRECISION *)threading->workspace)[0] = l->h_PRECISION.buffer2;
  ((vector_PRECISION *)threading->workspace)[1] = source;
  ((vector_PRECISION *)threading->workspace)[2] = l->h_PRECISION.buffer1;
  ((vector_PRECISION **)threading->workspace)[3] = l->h_PRECISION.X;
  ((vector_PRECISION *)threading->workspace)[4] = l->h_PRECISION.X[0];
  ((vector_PRECISION *)threading->workspace)[5] = l->h_PRECISION.sample;
  ((vector_PRECISION *)threading->workspace)[6] = l->h_PRECISION.total_variance;
  ((vector_PRECISION *)threading->workspace)[7] = l->h_PRECISION.rough_trace;
  //----------- Setting the pointers to each column of X
  for(i=0; i<block_size; i++)	l->h_PRECISION.X[i] = l->h_PRECISION.X[0]+i*l->inner_vector_size;
  END_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading)
  
  solution = ((vector_PRECISION *)threading->workspace)[0];
  source   = ((vector_PRECISION *)threading->workspace)[1];
  buffer   = ((vector_PRECISION *)threading->workspace)[2];
  X   = ((vector_PRECISION **)threading->workspace)[3];
	X[0] = ((vector_PRECISION *)threading->workspace)[4];
	sample = ((vector_PRECISION *)threading->workspace)[5];
	total_variance = ((vector_PRECISION *)threading->workspace)[6];
	rough_trace = ((vector_PRECISION *)threading->workspace)[7];

  //---------------------------------SET variances and estiamtes TO ZERO------------------------------------------------
	l->h_PRECISION.total_variance[0]=0.0; l->h_PRECISION.rough_trace[0]=0.0;
	memset(estimate, 0, sizeof(estimate));
	memset(variance, 0, sizeof(variance));
 
//----------------------------Compute ROUGH BLOCK TRACE----------------------------------------------------  
	gmres_PRECISION_struct* p = &(g.p);

  if(g.my_rank==0){
    START_MASTER(threading)
    printf("\t------computing rough trace ----------\n");
    END_MASTER(threading)
    fflush(stdout);
  }
	
	l->h_PRECISION.buffer_int1 = nr_rough_ests;
  block_hutchinson_PRECISION(l, threading, estimate, variance);

  if(g.my_rank==0){
    START_MASTER(threading)
    printf("\t------------- done--------------------\n");
    END_MASTER(threading)
    fflush(stdout);
  }
  

	START_MASTER(threading)
	for (i=0; i< block_size*block_size; i+=block_size+1)	rough_trace[0] += estimate[i]; 
  l->h_PRECISION.trace=0.0;
  
  END_MASTER(threading)
  fflush(stdout);
	SYNC_MASTER_TO_ALL(threading)




//---------------------------Compute BLOCK TRACE------------------------------------------------ 
 
  if(g.my_rank==0){
    START_MASTER(threading)
    printf("\t-------computing the block trace -----------\n");
    END_MASTER(threading)
    fflush(stdout);
  }

	
	l->h_PRECISION.total_variance[0]=0.0; l->h_PRECISION.trace=0.0; 
	memset(estimate, 0, sizeof(estimate));
	memset(variance, 0, sizeof(variance));

	l->h_PRECISION.buffer_int1 = nr_ests;
  block_hutchinson_PRECISION(l, threading, estimate, variance);
	
		
		
	if(g.my_rank==0){
    START_MASTER(threading)
    printf("\t------------------DONE----------------\n");
    END_MASTER(threading)
    fflush(stdout);
  }	
		
	trace= l->h_PRECISION.trace;	
	if(g.my_rank==0){
  START_MASTER(threading)
	printf( "Finally, Trace: %f + i%f \n ", CSPLIT(trace)  );
  END_MASTER(threading)
	SYNC_MASTER_TO_ALL(threading)
  fflush(stdout);
  }	



//------------------------SAVE BLOCK TRACE INTO A FILE-----------------------------------
 START_MASTER(threading)
	int j;		
 if(g.my_rank==0){		 
		char a[100] ;   
		sprintf(a, "%s%d%s", "BLOCK_TRACE",g.my_rank, ".txt");
		char fileSpec[strlen(a)+1];
		snprintf( fileSpec, sizeof( fileSpec ), "%s", a );  
		FILE * fp;  
		fp = fopen (fileSpec, "w+");
		
		sprintf(a, "%s%d%s", "BLOCK_Var",g.my_rank, ".txt");
		char fileSpec_var[strlen(a)+1];
		snprintf( fileSpec_var, sizeof( fileSpec_var ), "%s", a );  
		FILE * fvar;  
		fvar = fopen (fileSpec_var, "w+");
		
		for (i=0; i< block_size; i++){
			for(j=0; j<block_size; j++){
				fprintf(fp, "%f\t", creal(estimate[i*block_size+j]));
				fprintf(fvar, "%f\t", creal(variance[i*block_size+j]));
		 	}
		  fprintf(fp, "\n");
		  fprintf(fvar, "\n");
		}
		 		
		fclose(fp);		fclose(fvar);	
	}		 
	
	END_MASTER(threading)
	SYNC_MASTER_TO_ALL(threading)
 
  
  START_LOCKED_MASTER(threading)
  FREE( l->h_PRECISION.buffer2	, complex_PRECISION, l->inner_vector_size );
  FREE( source, complex_PRECISION, l->inner_vector_size );
  FREE(  l->h_PRECISION.buffer1, complex_PRECISION, l->inner_vector_size/block_size);

  FREE( l->h_PRECISION.X[0], complex_PRECISION, block_size* l->inner_vector_size);
  FREE( l->h_PRECISION.X, vector_PRECISION, block_size );
	FREE(l->h_PRECISION.sample, complex_PRECISION, block_size*block_size*nr_ests);
	FREE(total_variance, complex_PRECISION, 1);
	FREE(rough_trace, complex_PRECISION, 1);
  END_LOCKED_MASTER(threading)
		
  
}


/* Testing BIG X
  
		char a[100] ;    

    sprintf(a, "%s%d%s", "BIG",g.my_rank, ".txt");
    char fileSpec[strlen(a)+1];
    snprintf( fileSpec, sizeof( fileSpec ), "%s", a );

    
   FILE * fp;
   
   fp = fopen (fileSpec, "w+");
   
   
   for(j=0; j<block_size; j++){
		for(i=0; i<l->inner_vector_size; i++){
		   fprintf(fp, "%f\n",  creal(X[j][i]));
   //fprintf(fp, "%d \t %d\t %f ", j,i, creal(X[j][i]));
   }
      //fprintf(fp, "\n");
   }
   fclose(fp);
	
   
*/





void mlmc_hutchinson_rough_trace_PRECISION( level_struct *l, struct Thread *threading ) {
	
  
	int start, end, i, j, li;
	int nr_ests[g.num_levels];
	nr_ests[0]=10; 
	for(i=1; i< g.num_levels; i++) nr_ests[i] = nr_ests[i-1]*1;

  gmres_PRECISION_struct *p_PRECISION_finest = &(g.p);
  gmres_PRECISION_struct *ps[g.num_levels];
  level_struct *ls[g.num_levels];
  
  ps[0] = &(g.p);
  ls[0] = l;
  level_struct *lx = l->next_level;
  for( i=1;i<g.num_levels;i++ ){
    ps[i] = &(lx->p_PRECISION);
    ls[i] = lx;
    lx = l->next_level;
  }
  
	complex_PRECISION es[g.num_levels];
	memset( es,0,g.num_levels*sizeof(complex_PRECISION) );
	complex_PRECISION tmpx =0.0;
	vector_PRECISION mlmc_b1 = NULL;
		
  START_MASTER(threading)
  MALLOC( mlmc_b1, complex_PRECISION, l->inner_vector_size ); 
  ((vector_PRECISION *)threading->workspace)[4] = mlmc_b1;
  END_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading)
  mlmc_b1 = ((vector_PRECISION *)threading->workspace)[4];

	if(g.my_rank==0){
		START_MASTER(threading)
		printf("-------COMPUTING ROUGH_TRACE (%d Estimates)----------- \n\n",nr_ests[0] );
	 	END_MASTER(threading)
	}


  for( li=0;li<(g.num_levels-1);li++ ){

    compute_core_start_end( 0, ls[li]->inner_vector_size, &start, &end, l, threading );
		
		for(i=0;i<nr_ests[li]; i++){

		  START_MASTER(threading)
			vector_PRECISION_define_random_rademacher( ps[li]->b, 0, ls[li]->inner_vector_size, ls[li] );		
			END_MASTER(threading)
			SYNC_MASTER_TO_ALL(threading)
		 
			restrict_PRECISION( ps[li+1]->b, ps[li]->b, ls[li], threading ); // get rhs for next level.

			fgmres_PRECISION( ps[li+1], ls[li+1], threading );

			interpolate3_PRECISION( mlmc_b1, ps[li+1]->x, ls[li], threading ); //
			
			fgmres_PRECISION( ps[li], ls[li], threading );
			
			vector_PRECISION_minus( mlmc_b1, ps[li]->x, mlmc_b1, start, end, ls[li] );
			tmpx = global_inner_product_PRECISION( ps[li]->b, mlmc_b1, ps[li]->v_start, ps[li]->v_end, ls[li], threading );
		
			es[li] += tmpx;

			
		}
	}

  li = g.num_levels-1;
	for(i=0;i<nr_ests[li]; i++){
	  START_MASTER(threading)
		vector_PRECISION_define_random_rademacher( ps[li]->b, 0, ls[li]->inner_vector_size, ls[li] );		
		END_MASTER(threading)
		SYNC_MASTER_TO_ALL(threading)

		fgmres_PRECISION( ps[li], ls[li], threading );

	 	tmpx= global_inner_product_PRECISION( ps[li]->b, ps[li]->x, ps[li]->v_start, ps[li]->v_end, ls[li], threading );
	 
		es[li] += tmpx;

	}

	complex_PRECISION trace = 0.0;
	for( i=0;i<g.num_levels-1;i++ ){
	  trace += es[i]/nr_ests[i];
	}
	trace += es[g.num_levels-1]/nr_ests[i];
	
	START_MASTER(threading)
	l->h_PRECISION.mlmc_rough_trace=trace;
	END_MASTER(threading)	
	
	if(g.my_rank==0){
		START_MASTER(threading)
		printf("----------------Done-------------------\n");
		fflush(stdout);
		END_MASTER(threading)
	}

  START_MASTER(threading)
 	FREE( mlmc_b1, complex_PRECISION, l->inner_vector_size );
	END_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading)
}



void mlmc_hutchinson_diver_PRECISION( level_struct *l, struct Thread *threading ) {

  
  mlmc_hutchinson_rough_trace_PRECISION( l, threading );
  
printf("rough_trace %f\n", 	creal(l->h_PRECISION.mlmc_rough_trace));
	
	/*
  TODO :
  	 -- variance
  	 -- stopping criterium/condition
  */
}



/*

// 1 (Radamacher vectors of size of level 0)
A1 - P1 * A2^{-1} * R1

// 2 (Radamacher vectors of size of level 1)
P1 * A2^{-1} * R1 - P1 * P2 * A3^{-1} * R2 * R1
A2^{-1} - P2 * A3^{-1} * R2

// 3
P1 * P2 * A3^{-1} * R2 * R1 - P1 * P2 * P3 * A4^{-1} * R3 * R2 * R1
A3^{-1} - P3 * A4^{-1} * R3

// 4
P1 * P2 * P3 * A4^{-1} * R3 * R2 * R1
A4^{-1}

*/
