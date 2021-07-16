
#ifndef HUTCHINSON_PRECISION_HEADER
  #define HUTCHINSON_PRECISION_HEADER


  struct Thread;
  void block_hutchinson_driver_PRECISION( level_struct *l, struct Thread *threading );
  void hutchinson_driver_PRECISION( level_struct *l, struct Thread *threading );
  
  void mlmc_hutchinson_diver_PRECISION( vector_PRECISION phi, vector_PRECISION Dphi, vector_PRECISION eta,
                       int res, level_struct *l, struct Thread *threading );


#endif
