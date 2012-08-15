/*
 * Copyright 1997, Regents of the University of Minnesota
 * Modified by Jack Poulson, 2012
 */
#include "metislib.h"

void CliqOrder( ctrl_t *ctrl, graph_t *graph, idx_t *order, idx_t *sizes )
{
  idx_t i, nvtxs;
  idx_t offsets[3];
  idx_t *where;

  nvtxs = graph->nvtxs;

  MlevelNodeBisectionMultiple(ctrl, graph);

  IFSET(ctrl->dbglvl, METIS_DBG_SEPINFO, 
      printf("Nvtxs: %6"PRIDX", [%6"PRIDX" %6"PRIDX" %6"PRIDX"]\n", 
        graph->nvtxs, graph->pwgts[0], graph->pwgts[1], graph->pwgts[2]));

  /* Order the vertices */
  where = graph->where;
  sizes[0] = graph->pwgts[0];
  sizes[1] = graph->pwgts[1];
  sizes[2] = graph->pwgts[2];
  offsets[0] = 0;
  offsets[1] = sizes[0];
  offsets[2] = sizes[0]+sizes[1];
  for( i=0; i<nvtxs; ++i )
      order[i] = offsets[where[i]]++;

  FreeGraph(&graph);
}

/* TODO: Better error-handling */
void CliqBisect
( idx_t *nvtxs, idx_t *xadj, idx_t *adjncy,
  idx_t *numSeps, real_t *imbalance, idx_t *order, idx_t *sizes ) 
{
  int sigrval=0;
  idx_t options[METIS_NOPTIONS];
  graph_t *graph=NULL;
  ctrl_t *ctrl;

  if( *nvtxs == 0 )
  {
    sizes[0] = 0;
    sizes[1] = 0;
    sizes[2] = 0;
    return;
  }

  /* set up malloc cleaning code and signal catchers */
  if( !gk_malloc_init() )
    return;

  gk_sigtrap();

  if( (sigrval = gk_sigcatch()) != 0 ) 
    goto SIGTHROW;

  /* set up the run time parameters */
  METIS_SetDefaultOptions(options);
  options[METIS_OPTION_COMPRESS] = 0;
  options[METIS_OPTION_NSEPS] = *numSeps;
  options[METIS_OPTION_UFACTOR] = (int)((*imbalance-1)*1000);
  ctrl = SetupCtrl(METIS_OP_OMETIS, options, 1, 3, NULL, NULL);
  if (!ctrl) {
    gk_siguntrap();
    return;
  }

  IFSET(ctrl->dbglvl, METIS_DBG_TIME, InitTimers(ctrl));
  IFSET(ctrl->dbglvl, METIS_DBG_TIME, gk_startcputimer(ctrl->TotalTmr));

  graph = SetupGraph(ctrl, *nvtxs, 1, xadj, adjncy, NULL, NULL, NULL);

  ASSERT(CheckGraph(graph, 0, 1));

  /* allocate workspace memory */
  AllocateWorkSpace(ctrl, graph);

  /* compute the bisection ordering */
  CliqOrder( ctrl, graph, order, sizes );

  IFSET(ctrl->dbglvl, METIS_DBG_TIME, gk_stopcputimer(ctrl->TotalTmr));
  IFSET(ctrl->dbglvl, METIS_DBG_TIME, PrintTimers(ctrl));

  /* clean up */
  FreeCtrl(&ctrl);

SIGTHROW:
  gk_siguntrap();
  gk_malloc_cleanup(0);

  return;
}

