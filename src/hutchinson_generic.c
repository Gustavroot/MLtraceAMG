
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

void solve_PRECISION( vector_PRECISION solution, vector_PRECISION source, level_struct *l, struct Thread *threading, complex_PRECISION rt ){

  int iter = 0, start = threading->start_index[l->depth], end = threading->end_index[l->depth];

  vector_PRECISION rhs = g.mixed_precision==2?g.p_MP.dp.b:g.p.b;
  vector_PRECISION sol = g.mixed_precision==2?g.p_MP.dp.x:g.p.x;

  vector_PRECISION_copy( rhs, source, start, end, l );  
  iter = fgmres_PRECISION( &(g.p), l, threading );
  vector_PRECISION_copy( solution, sol, start, end, l );
}


complex_PRECISION hutchinson_driver_PRECISION( level_struct *l, struct Thread *threading ) {
  
  hutchinson_PRECISION_struct* h = &(l->h_PRECISION);
  vector_PRECISION solution = h->buffer1, source = h->buffer2;

  int i,j, nr_ests = h->max_iters;
  
  PRECISION trace_tol = l->h_PRECISION.trace_tol;
  complex_PRECISION trace=0.0;
  complex_PRECISION rough_trace=0.0;
  complex_PRECISION aux=0.0;
  complex_PRECISION sample[nr_ests];
  PRECISION variance = 0.0;
  PRECISION RMSD = 0.0;
  rough_trace = l->h_PRECISION.rt;
  
  aux=0.0; 
  gmres_PRECISION_struct* p = &(g.p); //g accesesible from any func.  
  for( i=0; i<nr_ests  ; i++ ) {

    g.if_rademacher = 1; //Compute random vector
    rhs_define_PRECISION( source, l, threading );
    g.if_rademacher = 0;

    solve_PRECISION( solution, source, l, threading, rough_trace ); //get A⁻¹x

    sample[i] = global_inner_product_PRECISION( source, solution, p->v_start, p->v_end, l, threading ); //compute x'A⁻¹x

    aux += sample[i];    trace = aux/ (i+1); //average the samples

    for (j=0; j<i; j++) // compute the variance
        variance += (  sample[j]- trace)*conj(  sample[j]- trace); 
    variance /= (j+1);

    RMSD = sqrt(variance/(j+1)); //RMSD= sqrt(var+ bias²)

    START_MASTER(threading)
    if(g.my_rank==0)  printf( "%d \t var %f \t RMSD %f < %f, \t Trace: %f + i%f \n ", i, variance, RMSD, cabs(rough_trace)*trace_tol, CSPLIT(trace)  );
    END_MASTER(threading)
    SYNC_MASTER_TO_ALL(threading)
    variance=0.0;
    if(i !=0 && RMSD< cabs(rough_trace)*trace_tol && i>=l->h_PRECISION.min_iters-1) break;
  }
  
  START_MASTER(threading)
  if(g.my_rank==0 && rough_trace!=0) printf( "\ntrace result = %f+i%f  for %d samples \n ",CSPLIT(trace), i );
  END_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading)
  
  return trace;
}


void block_hutchinson_PRECISION( level_struct *l, struct Thread *threading , complex_PRECISION* estimate, complex_PRECISION* variance ){
  vector_PRECISION solution = l->h_PRECISION.buffer2; vector_PRECISION* X = l->h_PRECISION.X;
  vector_PRECISION buffer = l->h_PRECISION.block_buffer; vector_PRECISION sample = l->h_PRECISION.sample;
  complex_PRECISION rough_trace = l->h_PRECISION.rough_trace;
  complex_PRECISION trace = l->h_PRECISION.trace;
  gmres_PRECISION_struct* p = &(g.p);
  int k, i, j, v, counter=0;
  int block_size=l->h_PRECISION.block_size, nr_ests=l->h_PRECISION.max_iters; 
  PRECISION trace_tol = l->h_PRECISION.trace_tol;
  
  for( k=0; k<nr_ests; k++ ) { //Hutchinson Method loop
    START_MASTER(threading)
    // Initialize buffer with Rademacher
    vector_PRECISION_define_random_rademacher( buffer, 0, l->inner_vector_size/block_size, l );    
    //---------------------- Fill Big X----------------------------------------------
    for(j=0; j<block_size; j++)
      for(i=j; i<l->inner_vector_size; i+=block_size)
           X[j][i] = buffer[i/12];           
    END_MASTER(threading)
    SYNC_MASTER_TO_ALL(threading)
    //-------------------------------------------------------------------------------
    
    //-----------------------COMPUTING THE BLOCK TRACE SAMPLE------------------------- 
    for(j=0; j<block_size; j++){  
      solve_PRECISION( solution, X[j], l, threading, 0.0 ); //get A⁻¹x 
      for (i=0; i< block_size; i++){
        complex_PRECISION tmpx = global_inner_product_PRECISION( X[i], solution, p->v_start, p->v_end, l, threading ); //compute x'A⁻¹x     
        START_MASTER(threading)
        sample[j+i*block_size+ k*block_size*block_size ] = tmpx;
        estimate[j+i*block_size] += sample[j+i*block_size+ k*block_size*block_size ];   
                               
        for (v=0; v<k; v++)  //compute the variance for the (i,j) element in the BT     
          variance[j+i*block_size] += (  sample[j+i*block_size+ v*block_size*block_size ]- estimate[j+i*block_size]/(k+1)  )*conj(  sample[j+i*block_size+ v*block_size*block_size ]- estimate[j+i*block_size]/(k+1) ); 
        variance[j+i*block_size] /= (v+1);  
                     
        END_MASTER(threading)      
        SYNC_MASTER_TO_ALL(threading)
      }          
    }   
    //----------------------------------------------------------------------------    
    
    //-----------------------COMPUTING Total Variation -------------------------
    START_MASTER(threading) 
    for (v=0; v< block_size*block_size; v++)  l->h_PRECISION.total_variance+= variance[v];     
    END_MASTER(threading)
    SYNC_MASTER_TO_ALL(threading)
    //---------------------------------------------------------------------------- 
          
    counter=k;
    if(k !=0 && sqrt(l->h_PRECISION.total_variance)/(k+1) < cabs(rough_trace)*trace_tol && k>=l->h_PRECISION.min_iters-1){counter=k; break; }

    START_MASTER(threading) 
    if(g.my_rank==0)
    printf( "%d \t RMSD  %f < %f ??\t first entry %f \t rough_trace %f\n", k, sqrt(l->h_PRECISION.total_variance)/(k+1), cabs(rough_trace)*trace_tol, creal(estimate[0])/(k+1), creal(rough_trace)  );
    END_MASTER(threading)
    SYNC_MASTER_TO_ALL(threading)
    
    l->h_PRECISION.total_variance=0.0;  //Reset Total Variation
  } //LOOP Hutchinson
  
  //-----------------------COMPUTING TRACE -------------------------  
  START_MASTER(threading)
  for (i=0; i< block_size*block_size; i++)  {estimate[i] /=(counter+1); } //Average each estimate    
  for (i=0; i< block_size*block_size; i+=block_size+1){  trace += estimate[i];} //Compute trace 
  l->h_PRECISION.trace=trace;
  END_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading)
  //---------------------------------------------------------------------------- 
}



void block_hutchinson_driver_PRECISION( level_struct *l, struct Thread *threading ) {
  START_MASTER(threading)
  if(g.my_rank==0)printf("Inside block_hutchinson_driver(...) ...\n");
  END_MASTER(threading)
 //----------------------------INITIALIZE VARIABLES---------------------------------------------------- 
  complex_PRECISION estimate[l->h_PRECISION.block_size*l->h_PRECISION.block_size] ;
  complex_PRECISION variance[l->h_PRECISION.block_size*l->h_PRECISION.block_size] ; 
  memset(estimate, 0, sizeof(estimate));
  memset(variance, 0, sizeof(variance));

  l->h_PRECISION.trace=0.0; l->h_PRECISION.rough_trace=0.0;
  l->h_PRECISION.total_variance=0.0; 
  
  int i;  
  

  //----------------------------Compute ROUGH block TRACE----------------------------------------------------  
  START_MASTER(threading)
  if(g.my_rank==0)printf("\t------computing ROUGH block trace ----------\n");
  END_MASTER(threading)

  l->h_PRECISION.max_iters = 5;  
  block_hutchinson_PRECISION(l, threading, estimate, variance);

  START_MASTER(threading)
  if(g.my_rank==0)printf("\t------------- done--------------------\n");
  END_MASTER(threading)

  START_MASTER(threading)
  for (i=0; i< l->h_PRECISION.block_size*l->h_PRECISION.block_size; i+=l->h_PRECISION.block_size+1)  l->h_PRECISION.rough_trace += estimate[i];  
  END_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading)

//---------------------------Compute BLOCK TRACE------------------------------------------------ 
  START_MASTER(threading)
   if(g.my_rank==0) printf("\t-------computing the block trace -----------\n");
  END_MASTER(threading)
 
  l->h_PRECISION.total_variance=0.0; l->h_PRECISION.trace=0.0; 
  memset(estimate, 0, sizeof(estimate));
  memset(variance, 0, sizeof(variance));

  l->h_PRECISION.max_iters=1000;
  block_hutchinson_PRECISION(l, threading, estimate, variance);
  
  START_MASTER(threading)
  if(g.my_rank==0)printf("\t------------------DONE----------------\n");
  END_MASTER(threading)

  START_MASTER(threading)
  if(g.my_rank==0)printf( "Finally, Trace: %f + i%f \n ", CSPLIT(l->h_PRECISION.trace)  );
  END_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading)

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
    
    for (i=0; i< l->h_PRECISION.block_size; i++){
      for(j=0; j<l->h_PRECISION.block_size; j++){
        fprintf(fp, "%f\t", creal(estimate[i*l->h_PRECISION.block_size+j]));
        fprintf(fvar, "%f\t", creal(variance[i*l->h_PRECISION.block_size+j]));
       }
      fprintf(fp, "\n");
      fprintf(fvar, "\n");
    }
         
    fclose(fp);    fclose(fvar);  
  }     
  
  END_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading) 
}


/* Testing BIG X
  START_MASTER(threading)
  if(g.my_rank==0){  
    char a[100] ;    
    sprintf(a, "%s%d%s", "BIG",g.my_rank, ".txt");
    char fileSpec[strlen(a)+1];
    snprintf( fileSpec, sizeof( fileSpec ), "%s", a );
       
   FILE * fp;  
   fp = fopen (fileSpec, "w+");
      
   for(j=0; j<l->h_PRECISION.block_size; j++){
    for(i=0; i<l->inner_vector_size; i++){
       fprintf(fp, "%f\n",  creal(X[j][i]));
   //fprintf(fp, "%d \t %d\t %f ", j,i, creal(X[j][i]));
   }
      //fprintf(fp, "\n");
   }
   fclose(fp);  
*/



void hutchinson_diver_PRECISION_init( level_struct *l, struct Thread *threading ) {
  hutchinson_PRECISION_struct* h = &(l->h_PRECISION);

//PLAIN  
  h->buffer1 = NULL; //Solution
  h->buffer2 = NULL; //Source
//MLMC
  h->mlmc_b1 = NULL;
  h->mlmc_b1_float=NULL;
  
//BLOCK  
  h->block_buffer = NULL;
  h->X = NULL;            
  h->sample  = NULL;
  
  h->X_float = NULL;  
   
  SYNC_MASTER_TO_ALL(threading)
}

void hutchinson_diver_PRECISION_alloc( level_struct *l, struct Thread *threading ) {
  hutchinson_PRECISION_struct* h = &(l->h_PRECISION) ;

//For PLAIN hutchinson
  START_MASTER(threading)
  MALLOC(h->buffer1, complex_PRECISION, l->inner_vector_size ); //solution
  MALLOC(h->buffer2, complex_PRECISION, l->inner_vector_size ); // source
  ((vector_PRECISION *)threading->workspace)[0] = h->buffer1;
  ((vector_PRECISION *)threading->workspace)[1] = h->buffer2;

//For MLMC
  MALLOC( h->mlmc_b1, complex_PRECISION, l->inner_vector_size );   
  MALLOC( h->mlmc_b1_float, complex_float, l->inner_vector_size );  
  ((vector_PRECISION *)threading->workspace)[2] = h->mlmc_b1; 
  ((vector_float *)threading->workspace)[3] = h->mlmc_b1_float;  

//for BLOCK
  MALLOC( h->block_buffer, complex_PRECISION, l->inner_vector_size/h->block_size );
  MALLOC( h->X, vector_PRECISION, h->block_size );
  MALLOC( h->X[0], complex_PRECISION, h->block_size* l->inner_vector_size);
  MALLOC( h->sample, complex_PRECISION, h->block_size*h->block_size*h->max_iters);
  ((vector_PRECISION *)threading->workspace)[4] = h->block_buffer;
  ((vector_PRECISION **)threading->workspace)[5] = h->X;	
  ((vector_PRECISION *)threading->workspace)[6] = h->X[0];
  ((vector_PRECISION *)threading->workspace)[7] = h->sample;

//for BLOCK mlmc  
  MALLOC( h->X_float, vector_float, h->block_size );//for mlmc
  MALLOC( h->X_float[0], complex_float, h->block_size* l->inner_vector_size);//for mlmc
  ((vector_float **)threading->workspace)[8] = h->X_float; //for mlmc
  ((vector_float *)threading->workspace)[9] = h->X_float[0]; //for mlmc 
  END_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading)

  h->buffer1 = ((vector_PRECISION *)threading->workspace)[0];
  h->buffer2 = ((vector_PRECISION *)threading->workspace)[1];
  h->mlmc_b1  = ((vector_PRECISION *)threading->workspace)[2];
  h->mlmc_b1_float = ((vector_float *)threading->workspace)[3];
  h->block_buffer =  ((vector_PRECISION *)threading->workspace)[4] ;
  h->X = ((vector_PRECISION **)threading->workspace)[5] ;
  h->X[0] = ((vector_PRECISION *)threading->workspace)[6] ;
  h->sample = ((vector_PRECISION *)threading->workspace)[7] ;
  h->X_float = ((vector_float **)threading->workspace)[8] ; //for mlmc
  h->X_float[0] = ((vector_float *)threading->workspace)[9] ; //for mlmc
	  
  //TODO: MOVE TO ALLOC???
  //----------- Setting the pointers to each column of X
  START_MASTER(threading)
  for(int i=0; i<l->h_PRECISION.block_size; i++)  l->h_PRECISION.X[i] = l->h_PRECISION.X[0]+i*l->inner_vector_size;
  for(int i=0; i<l->h_PRECISION.block_size; i++)  l->h_PRECISION.X_float[i] = l->h_PRECISION.X_float[0]+i*l->inner_vector_size;
  END_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading)
}

void hutchinson_diver_PRECISION_free( level_struct *l, struct Thread *threading ) {
  hutchinson_PRECISION_struct* h = &(l->h_PRECISION) ;

  START_MASTER(threading)
  FREE( h->mlmc_b1, complex_PRECISION, l->inner_vector_size );   
  FREE( h->mlmc_b1_float, complex_float, l->inner_vector_size );   
  FREE(h->buffer1, complex_PRECISION, l->inner_vector_size ); //solution
  FREE(h->buffer2, complex_PRECISION, l->inner_vector_size ); // source
  
  FREE( h->block_buffer, complex_PRECISION, l->inner_vector_size/h->block_size );
  FREE( h->X[0], complex_PRECISION, h->block_size* l->inner_vector_size);
  FREE( h->X, vector_PRECISION, h->block_size );

  FREE( h->sample, complex_PRECISION, h->block_size*h->block_size*h->max_iters);

  //FREE( h->X_float[0], complex_float, h->block_size* l->inner_vector_size);//for mlmc  
  FREE( h->X_float, vector_float, h->block_size );//for mlmc


  END_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading)
}




complex_PRECISION mlmc_hutchinson_diver_PRECISION( level_struct *l, struct Thread *threading ) {
    
  hutchinson_PRECISION_struct* h = &(l->h_PRECISION) ;
  complex_PRECISION rough_trace = h->rt;
  complex_PRECISION trace=0.0;
//--------------Setting the tolerance per level----------------
  int i;
  PRECISION est_tol = l->h_PRECISION.trace_tol;//  1E-6;
  PRECISION delta[g.num_levels];
  PRECISION tol[g.num_levels];
  PRECISION d0 = 0.4; 
  delta[0] = d0; tol[0] = sqrt(delta[0]);
  for(i=1 ;i<g.num_levels; i++){
    delta[i] = (1.0-d0)/(g.num_levels-1);
    tol[i] = sqrt(delta[i]);
  }
//-------------------------------------------------------------  

//-----------------Setting variables in levels-----------------  
  int start, end, j, li;
  int nr_ests[g.num_levels];
  int counter[g.num_levels];
  PRECISION RMSD;
  PRECISION variance=0.0;  
  nr_ests[0]=l->h_PRECISION.max_iters; 
  for(i=1; i< g.num_levels; i++) nr_ests[i] = nr_ests[i-1]*1; //TODO: Change in every level?
  
  gmres_float_struct *ps[g.num_levels]; //float p_structs for coarser levels
  gmres_double_struct *ps_double[1];   //double p_struct for finest level
  level_struct *ls[g.num_levels];  
  ps_double[0] = &(g.p);
  ls[0] = l;
  
  level_struct *lx = l->next_level;
  double buff_tol[g.num_levels];
  buff_tol[0] =  ps_double[i]->tol; 
  START_MASTER(threading)
  if(g.my_rank==0) printf("LEVEL----------- \t %d  \t vector size: %d \t TOLERANCE: %e\n", 0, ls[0]->inner_vector_size, buff_tol[0]);
  END_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading)
  for( i=1;i<g.num_levels;i++ ){
    ps[i] = &(lx->p_float); // p_PRECISION in each level
    ls[i] = lx;                  // level_struct of each level
    lx = lx->next_level;
    buff_tol[i]= ps[i]->tol;
    SYNC_MASTER_TO_ALL(threading)
    START_MASTER(threading)
    if(g.my_rank==0) printf("LEVEL----------- \t %d  \t vector size: %d \tTOLERANCE: %f\n ", i, ls[i]->inner_vector_size, buff_tol[i]);
    END_MASTER(threading)
          
  }
    
  complex_PRECISION es[g.num_levels];  //An estimate per level
  memset( es,0,g.num_levels*sizeof(complex_PRECISION) );
  complex_PRECISION tmpx =0.0;
//---------------------------------------------------------------------  

//-----------------Hutchinson for all but coarsest level-----------------    
  for( li=0;li<(g.num_levels-1);li++ ){
             
    variance=0.0;    counter[li]=0;
    complex_PRECISION sample[nr_ests[li]]; 
    compute_core_start_end( 0, ls[li]->inner_vector_size, &start, &end, l, threading );
    
    START_MASTER(threading)
    if(g.my_rank==0) printf("LEVEL----------- \t %d  \t ests: %d  \t start: %d  \t end: %d \t vector size: %d \n ", li, nr_ests[li], start, ps_double[li]->v_end, ls[li]->inner_vector_size);
    END_MASTER(threading)
    
    for(i=0;i<nr_ests[li]  ; i++){
      START_MASTER(threading) //Get random vector:
      if(li==0){
        vector_double_define_random_rademacher( ps_double[li]->b, 0, ls[li]->inner_vector_size, ls[li] );          
      }
      else
        vector_float_define_random_rademacher( ps[li]->b, 0, ls[li]->inner_vector_size, ls[li] );    
      END_MASTER(threading)
      SYNC_MASTER_TO_ALL(threading)
      //-----------------Solve the system in current level and the next one----------------- 
      if(li==0){
        trans_float( l->sbuf_float[0], ps_double[li]->b, l->s_float.op.translation_table, l, threading );     
        restrict_float( ps[li+1]->b, l->sbuf_float[0], ls[li], threading ); // get rhs for next level.
        fgmres_float( ps[li+1], ls[li+1], threading );               //solve in next level
        interpolate3_float( l->sbuf_float[1], ps[li+1]->x, ls[li], threading ); //      
        trans_back_float( h->mlmc_b1, l->sbuf_float[1], l->s_float.op.translation_table, l, threading );
        fgmres_double( ps_double[li], ls[li], threading );    
      }
      else{
        restrict_float( ps[li+1]->b, ps[li]->b, ls[li], threading ); // get rhs for next level.
        ps[li+1]->tol = 1e-7;
        fgmres_float( ps[li+1], ls[li+1], threading );               //solve in next level
        ps[li+1]->tol = buff_tol[li+1];
        interpolate3_float( h->mlmc_b1_float, ps[li+1]->x, ls[li], threading ); //   
        ps[li]->tol = 1e-7;   
        fgmres_float( ps[li], ls[li], threading );
        ps[li+1]->tol = buff_tol[li+1];
      }     
      //------------------------------------------------------------------------------------
      
      //--------------Get the difference of the solutions and the corresponding sample--------------                        
      if(li==0){     
        vector_double_minus( h->mlmc_b1, ps_double[li]->x, h->mlmc_b1, start, end, ls[li] );
        tmpx = global_inner_product_PRECISION( ps_double[li]->b, h->mlmc_b1, ps_double[li]->v_start, ps_double[li]->v_end, ls[li], threading );      
        sample[i]= tmpx;      
        es[li] += tmpx;
      }
      else{
        vector_float_minus( h->mlmc_b1_float, ps[li]->x, h->mlmc_b1_float, start, end, ls[li] );
        tmpx =  global_inner_product_float( ps[li]->b, h->mlmc_b1_float, ps[li]->v_start, ps[li]->v_end, ls[li], threading );      
        sample[i]= tmpx;      
        es[li] +=  tmpx;
      }
     //----------------------------------------------------------------------------------------------
     
     //--------------Get the Variance in the current level and use it in stop criteria---------------                              
      for (j=0; j<i; j++) // compute the variance
        variance += (sample[j]- es[li]/(i+1)) *conj( sample[j]- es[li]/(i+1)); 
      variance /= (j+1);
      RMSD = sqrt(variance/(j+1)); //RMSD= sqrt(var+ bias²)
     
      START_MASTER(threading)
      if(g.my_rank==0)
      printf( "%d \t var %f \t Est %f  \t RMSD %f <  %f ?? \n ", i, variance, creal( es[li]/(i+1) ), RMSD,  cabs(rough_trace)*tol[li]*est_tol );
      END_MASTER(threading)
           
      counter[li]=i+1;
      if(i !=0 && RMSD< cabs(rough_trace)*tol[li]*est_tol && i>=l->h_PRECISION.min_iters-1){counter[li]=i+1; break;}
      //----------------------------------------------------------------------------------------------    
    }
  }
  

  SYNC_MASTER_TO_ALL(threading)
//-----------------Hutchinson for just coarsest level-----------------       
  li = g.num_levels-1;
  
  START_MASTER(threading)
  if(g.my_rank==0)printf("LEVEL----------- \t %d  \t ests: %d  \t start: %d  \t end: %d \t vector size: %d \n ", li, nr_ests[li], start, ps[li]->v_end, ls[li]->inner_vector_size);
  END_MASTER(threading)
  
  complex_PRECISION sample[nr_ests[li]]; 
  variance = 0.0;
  counter[li]=0;
  for(i=0;i<nr_ests[li]  ; i++){
    START_MASTER(threading)
    vector_float_define_random_rademacher( ps[li]->b, 0, ls[li]->inner_vector_size, ls[li] );    
    END_MASTER(threading)
    SYNC_MASTER_TO_ALL(threading)
    
    ps[li]->tol = 1e-7;
    fgmres_float( ps[li], ls[li], threading );
    ps[li]->tol = buff_tol[li];
    tmpx= global_inner_product_float( ps[li]->b, ps[li]->x, ps[li]->v_start, ps[li]->v_end, ls[li], threading );   
    sample[i]= tmpx;      
    es[li] += tmpx;
  
    for (j=0; j<i; j++) // compute the variance
      variance += (sample[j]- es[li]/(i+1)) *conj( sample[j]- es[li]/(i+1)); 
    variance /= (j+1);
    RMSD = sqrt(variance/(j+1)); //RMSD= sqrt(var+ bias²)
        
    START_MASTER(threading)
    if(g.my_rank==0)
      printf( "%d \t var %f \t Est %f  \t RMSD %f <  %f ?? \n ", i, variance, creal( es[li]/(i+1) ), RMSD,  cabs(rough_trace)*tol[li]*est_tol );
    END_MASTER(threading)
    
    counter[li]=i+1;
    if(i !=0 && RMSD< cabs(rough_trace)*tol[li]*est_tol && i>=l->h_PRECISION.min_iters-1 ){counter[li]=i+1; break;}
  }


  trace = 0.0;
  for( i=0;i<g.num_levels;i++ ){
    trace += es[i]/counter[i];    
  START_MASTER(threading)
  if(g.my_rank==0)printf("Level:%d, ................................%f    %d\n", i, creal(es[i]),counter[i]);
  END_MASTER(threading)  

  }
  
  START_MASTER(threading)
  if(g.my_rank==0)
    printf( "\n\n\n\n %f+i %f \n", CSPLIT( trace ));
  END_MASTER(threading)
  
  
  return trace;
  
}







































complex_PRECISION mlmc_block_hutchinson_diver_PRECISION( level_struct *l, struct Thread *threading ) {
    
  hutchinson_PRECISION_struct* h = &(l->h_PRECISION) ;
  complex_PRECISION rough_trace = h->rt;
  complex_PRECISION trace=0.0;
//--------------Setting the tolerance per level----------------
  int i;
  PRECISION est_tol = l->h_PRECISION.trace_tol;//  1E-6;
  PRECISION delta[g.num_levels];
  PRECISION tol[g.num_levels];
  PRECISION d0 = 0.4; 
  delta[0] = d0; tol[0] = sqrt(delta[0]);
  for(i=1 ;i<g.num_levels; i++){
    delta[i] = (1.0-d0)/(g.num_levels-1);
    tol[i] = sqrt(delta[i]);
  }
//-------------------------------------------------------------  

//-----------------Setting variables in levels-----------------  
  int start, end, j, li;
  int nr_ests[g.num_levels];
  int counter[g.num_levels];
  PRECISION RMSD;
  PRECISION variance=0.0;  
  nr_ests[0]=l->h_PRECISION.max_iters; 
  for(i=1; i< g.num_levels; i++) nr_ests[i] = nr_ests[i-1]*1; //TODO: Change in every level?
  
  gmres_float_struct *ps[g.num_levels]; //float p_structs for coarser levels
  gmres_double_struct *ps_double[1];   //double p_struct for finest level
  level_struct *ls[g.num_levels];  
  ps_double[0] = &(g.p);
  ls[0] = l;
  
  level_struct *lx = l->next_level;
  double buff_tol[g.num_levels];
  buff_tol[0] =  ps_double[i]->tol; 
  START_MASTER(threading)
  if(g.my_rank==0) printf("LEVEL----------- \t %d  \t vector size: %d \t TOLERANCE: %e\n", 0, ls[0]->inner_vector_size, buff_tol[0]);
  END_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading)
  for( i=1;i<g.num_levels;i++ ){
    ps[i] = &(lx->p_float); // p_PRECISION in each level
    ls[i] = lx;                  // level_struct of each level
    lx = lx->next_level;
    buff_tol[i]= ps[i]->tol;
    SYNC_MASTER_TO_ALL(threading)
    START_MASTER(threading)
    if(g.my_rank==0) printf("LEVEL----------- \t %d  \t vector size: %d \tTOLERANCE: %f\n ", i, ls[i]->inner_vector_size, buff_tol[i]);
    END_MASTER(threading)
          
  }
    
  complex_PRECISION es[g.num_levels];  //An estimate per level
  memset( es,0,g.num_levels*sizeof(complex_PRECISION) );
  complex_PRECISION tmpx =0.0;
//---------------------------------------------------------------------  
      int block_size=h->block_size, row=0, col=0;
      vector_double* X = ls[0]->h_double.X;
      vector_float* X_float = ls[0]->h_double.X_float;
//-----------------Hutchinson for all but coarsest level-----------------    
  for( li=0;li<(g.num_levels-1);li++ ){
    
           
    variance=0.0;    counter[li]=0;
    complex_PRECISION sample[nr_ests[li]]; 
    compute_core_start_end( 0, ls[li]->inner_vector_size, &start, &end, l, threading );
    
    START_MASTER(threading)
    if(g.my_rank==0) printf("LEVEL----------- \t %d  \t ests: %d  \t start: %d  \t end: %d \t vector size: %d \n ", li, nr_ests[li], start, ps_double[li]->v_end, ls[li]->inner_vector_size);
    END_MASTER(threading)
    
    for(i=0;i<nr_ests[li] ; i++){
      START_MASTER(threading) //Get random vector:
      if(li==0){
        vector_double_define_random_rademacher( ps_double[li]->b, 0, ls[li]->inner_vector_size/block_size, ls[li] );
        //---------------------- Fill Big X----------------------------------------------      
		    for(col=0; col<block_size; col++)
		      for(row=col; row<ls[li]->inner_vector_size; row+=block_size)
		        X[col][row] = ps_double[li]->b[row/12];       
      }
      else{
        vector_float_define_random_rademacher( ps[li]->b, 0, ls[li]->inner_vector_size/block_size, ls[li] );
        //---------------------- Fill Big X----------------------------------------------      
		    for(col=0; col<block_size; col++)
		      for(row=col; row<ls[li]->inner_vector_size; row+=block_size)
		        X_float[col][row] = ps[li]->b[row/12];    
      }    
      END_MASTER(threading)
      SYNC_MASTER_TO_ALL(threading)

      //-----------------Solve the system in current level and the next one-----------------
      for(col=0; col<block_size; col++){
		    if(li==0){
		      trans_float( l->sbuf_float[0], X[col], l->s_float.op.translation_table, l, threading );     
		      restrict_float( ps[li+1]->b, l->sbuf_float[0], ls[li], threading ); // get rhs for next level.
		      fgmres_float( ps[li+1], ls[li+1], threading );               //solve in next level
		      interpolate3_float( l->sbuf_float[1], ps[li+1]->x, ls[li], threading ); //      
		      trans_back_float( X[col], l->sbuf_float[1], l->s_float.op.translation_table, l, threading );
		      fgmres_double( ps_double[li], ls[li], threading );    
		    }
     
      else{
        restrict_float( ps[li+1]->b, X_float[col], ls[li], threading ); // get rhs for next level.
        ps[li+1]->tol = 1e-7;
        fgmres_float( ps[li+1], ls[li+1], threading );               //solve in next level
        ps[li+1]->tol = buff_tol[li+1];
        interpolate3_float( h->mlmc_b1_float, ps[li+1]->x, ls[li], threading ); //   
        ps[li]->tol = 1e-7;   
        fgmres_float( ps[li], ls[li], threading );
        ps[li+1]->tol = buff_tol[li+1];
      }
       }//loop for col 
     }//temporal for i
     }//temporal for li
      /*     
      //------------------------------------------------------------------------------------
      
      //--------------Get the difference of the solutions and the corresponding sample--------------                        
      if(li==0){     
        vector_double_minus( h->mlmc_b1, ps_double[li]->x, h->mlmc_b1, start, end, ls[li] );
        tmpx = global_inner_product_PRECISION( ps_double[li]->b, h->mlmc_b1, ps_double[li]->v_start, ps_double[li]->v_end, ls[li], threading );      
        sample[i]= tmpx;      
        es[li] += tmpx;
      }
      else{
        vector_float_minus( h->mlmc_b1_float, ps[li]->x, h->mlmc_b1_float, start, end, ls[li] );
        tmpx =  global_inner_product_float( ps[li]->b, h->mlmc_b1_float, ps[li]->v_start, ps[li]->v_end, ls[li], threading );      
        sample[i]= tmpx;      
        es[li] +=  tmpx;
      }
     //----------------------------------------------------------------------------------------------
     
     //--------------Get the Variance in the current level and use it in stop criteria---------------                              
      for (j=0; j<i; j++) // compute the variance
        variance += (sample[j]- es[li]/(i+1)) *conj( sample[j]- es[li]/(i+1)); 
      variance /= (j+1);
      RMSD = sqrt(variance/(j+1)); //RMSD= sqrt(var+ bias²)
     
      START_MASTER(threading)
      if(g.my_rank==0)
      printf( "%d \t var %f \t Est %f  \t RMSD %f <  %f ?? \n ", i, variance, creal( es[li]/(i+1) ), RMSD,  cabs(rough_trace)*tol[li]*est_tol );
      END_MASTER(threading)
           
      counter[li]=i+1;
      if(i !=0 && RMSD< cabs(rough_trace)*tol[li]*est_tol && i>=l->h_PRECISION.min_iters-1){counter[li]=i+1; break;}
      //----------------------------------------------------------------------------------------------    
    }
  }
*/  

  SYNC_MASTER_TO_ALL(threading)
//-----------------Hutchinson for just coarsest level-----------------       
  li = g.num_levels-1;
  
  START_MASTER(threading)
  if(g.my_rank==0)printf("LEVEL----------- \t %d  \t ests: %d  \t start: %d  \t end: %d \t vector size: %d \n ", li, nr_ests[li], start, ps[li]->v_end, ls[li]->inner_vector_size);
  END_MASTER(threading)
  
  complex_PRECISION sample[nr_ests[li]]; 
  variance = 0.0;
  counter[li]=0;
  for(i=0;i<nr_ests[li]  ; i++){
    START_MASTER(threading)
    vector_float_define_random_rademacher( ps[li]->b, 0, ls[li]->inner_vector_size, ls[li] );          
    //---------------------- Fill Big X----------------------------------------------      
    for(col=0; col<block_size; col++){
		    for(row=col; row<ls[li]->inner_vector_size; row+=block_size)
		         ls[0]->h_double.X[col][row] = ps[li]->b[row/12];    
		         		       if(g.my_rank==0) printf("\t level %d\t col %d \t row: %d \n ", li, col, row);}
    
    END_MASTER(threading)
    SYNC_MASTER_TO_ALL(threading)
/*    
    ps[li]->tol = 1e-7;
    fgmres_float( ps[li], ls[li], threading );
    ps[li]->tol = buff_tol[li];
    tmpx= global_inner_product_float( ps[li]->b, ps[li]->x, ps[li]->v_start, ps[li]->v_end, ls[li], threading );   
    sample[i]= tmpx;      
    es[li] += tmpx;
  }//temporal bra for i loop
 
    for (j=0; j<i; j++) // compute the variance
      variance += (sample[j]- es[li]/(i+1)) *conj( sample[j]- es[li]/(i+1)); 
    variance /= (j+1);
    RMSD = sqrt(variance/(j+1)); //RMSD= sqrt(var+ bias²)
        
    START_MASTER(threading)
    if(g.my_rank==0)
      printf( "%d \t var %f \t Est %f  \t RMSD %f <  %f ?? \n ", i, variance, creal( es[li]/(i+1) ), RMSD,  cabs(rough_trace)*tol[li]*est_tol );
    END_MASTER(threading)
    
    counter[li]=i+1;
    if(i !=0 && RMSD< cabs(rough_trace)*tol[li]*est_tol && i>=l->h_PRECISION.min_iters-1 ){counter[li]=i+1; break;}
  }


  trace = 0.0;
  for( i=0;i<g.num_levels;i++ ){
    trace += es[i]/counter[i];    
  START_MASTER(threading)
  if(g.my_rank==0)printf("Level:%d, ................................%f    %d\n", i, creal(es[i]),counter[i]);
  END_MASTER(threading)  

  }
*/

}//temporal for i//
 
 
  
  
  
  START_MASTER(threading)
  if(g.my_rank==0)
    printf( "\n\n\n\n %f+i %f \n", CSPLIT( trace ));
  END_MASTER(threading)

  
  return trace;
  
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
