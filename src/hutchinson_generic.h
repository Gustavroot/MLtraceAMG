
#ifndef HUTCHINSON_PRECISION_HEADER
  #define HUTCHINSON_PRECISION_HEADER


  struct Thread;
  void block_hutchinson_driver_PRECISION( level_struct *l, struct Thread *threading );
  complex_PRECISION hutchinson_driver_PRECISION( level_struct *l, struct Thread *threading );
  void mlmc_hutchinson_rough_trace_PRECISION( level_struct *l, struct Thread *threading);
  complex_PRECISION mlmc_hutchinson_diver_PRECISION( level_struct *l, struct Thread *threading );
	
	void mlmc_hutchinson_diver_PRECISION_init( level_struct *l);
	void mlmc_hutchinson_diver_PRECISION_alloc( level_struct *l );
	void mlmc_hutchinson_diver_PRECISION_free( level_struct *l);

#endif
