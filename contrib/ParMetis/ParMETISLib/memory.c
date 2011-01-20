/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * memory.c
 *
 * This file contains routines that deal with memory allocation
 *
 * Started 2/24/96
 * George
 *
 * $Id: memory.c,v 1.3 2003/07/30 18:37:59 karypis Exp $
 *
 */

#include <parmetislib.h>


/*************************************************************************/
/*! This function allocate various pools of memory */
/*************************************************************************/
void AllocateWSpace(CtrlType *ctrl, GraphType *graph, WorkSpaceType *wspace)
{
  wspace->nlarge  = 2*graph->nedges;
  wspace->nparts  = ctrl->nparts;
  wspace->npes    = ctrl->npes;

  wspace->maxcore = 8*graph->nedges+1;
  wspace->core    = idxmalloc(wspace->maxcore, "AllocateWSpace: wspace->core");

  wspace->pairs   = (KeyValueType *)wspace->core;
  wspace->indices = (idxtype *)(wspace->pairs + wspace->nlarge);
  wspace->degrees = (EdgeType *)(wspace->indices + wspace->nlarge);


  wspace->pv1 = idxmalloc(ctrl->nparts+ctrl->npes+1, "AllocateWSpace: wspace->pv1");
  wspace->pv2 = idxmalloc(ctrl->nparts+ctrl->npes+1, "AllocateWSpace: wspace->pv2");
  wspace->pv3 = idxmalloc(ctrl->nparts+ctrl->npes+1, "AllocateWSpace: wspace->pv3");
  wspace->pv4 = idxmalloc(ctrl->nparts+ctrl->npes+1, "AllocateWSpace: wspace->pv4");

  wspace->pepairs1 = (KeyValueType *)GKmalloc(sizeof(KeyValueType)*(ctrl->nparts+ctrl->npes+1), "AllocateWSpace: wspace->pepairs?");
  wspace->pepairs2 = (KeyValueType *)GKmalloc(sizeof(KeyValueType)*(ctrl->nparts+ctrl->npes+1), "AllocateWSpace: wspace->pepairs?");

}

/*************************************************************************/
/*! This function re-allocates the workspace if previous one is not large
    enough */
/*************************************************************************/
void AdjustWSpace(CtrlType *ctrl, GraphType *graph, WorkSpaceType *wspace)
{
  if (wspace->nlarge < 2*graph->nedges || wspace->nparts < ctrl->nparts || wspace->npes < ctrl->npes) {
    FreeWSpace(wspace);
    AllocateWSpace(ctrl, graph, wspace);
  }
}

/*************************************************************************/
/*! This function de-allocate various pools of memory */
/**************************************************************************/
void FreeWSpace(WorkSpaceType *wspace)
{
  GKfree((void **)&wspace->core);
  GKfree((void **)&wspace->pv1);
  GKfree((void **)&wspace->pv2);
  GKfree((void **)&wspace->pv3);
  GKfree((void **)&wspace->pv4);
  GKfree((void **)&wspace->pepairs1);
  GKfree((void **)&wspace->pepairs2);
}


/*************************************************************************
* This function de-allocates memory allocated for the control structures
**************************************************************************/
void FreeCtrl(CtrlType *ctrl)
{
  MPI_Comm_free(&(ctrl->gcomm));
}


/*************************************************************************
* This function creates a CoarseGraphType data structure and initializes
* the various fields
**************************************************************************/
GraphType *CreateGraph(void)
{
  GraphType *graph;

  graph = (GraphType *)GKmalloc(sizeof(GraphType), "CreateCoarseGraph: graph");

  InitGraph(graph);

  return graph;
}


/*************************************************************************
* This function creates a CoarseGraphType data structure and initializes
* the various fields
**************************************************************************/
void InitGraph(GraphType *graph) 
{
  graph->gnvtxs = graph->nvtxs = graph->nedges = graph->nsep = -1;
  graph->nnbrs = graph->nrecv = graph->nsend = graph->nlocal = -1;
  graph->xadj = graph->vwgt = graph->vsize = graph->adjncy = graph->adjwgt = NULL;
  graph->nvwgt = NULL;
  graph->vtxdist = NULL;
  graph->match = graph->cmap = NULL;
  graph->label = NULL;

  graph->peind = NULL;
  graph->sendptr = graph->sendind = graph->recvptr = graph->recvind = NULL;
  graph->imap = NULL;
  graph->pexadj = graph->peadjncy = graph->peadjloc = NULL;
  graph->lperm = NULL;

  graph->slens = graph->rlens = NULL;
  graph->rcand = NULL;

  graph->where = graph->home = graph->lpwgts = graph->gpwgts = NULL;
  graph->lnpwgts = graph->gnpwgts = NULL;
  graph->rinfo = NULL;

  graph->nrinfo  = NULL;
  graph->sepind  = NULL;
  graph->hmarker = NULL;

  graph->coarser = graph->finer = NULL;

}

/*************************************************************************
* This function deallocates any memory stored in a graph
**************************************************************************/
void FreeGraph(GraphType *graph) 
{

  GKfree((void **)&graph->xadj);
  GKfree((void **)&graph->vwgt);
  GKfree((void **)&graph->nvwgt);
  GKfree((void **)&graph->vsize);
  GKfree((void **)&graph->adjncy);
  GKfree((void **)&graph->adjwgt);
  GKfree((void **)&graph->vtxdist);
  GKfree((void **)&graph->match);
  GKfree((void **)&graph->cmap);
  GKfree((void **)&graph->lperm);
  GKfree((void **)&graph->label);
  GKfree((void **)&graph->where);
  GKfree((void **)&graph->home);
  GKfree((void **)&graph->rinfo);
  GKfree((void **)&graph->nrinfo);
  GKfree((void **)&graph->sepind);
  GKfree((void **)&graph->hmarker);
  GKfree((void **)&graph->lpwgts);
  GKfree((void **)&graph->gpwgts);
  GKfree((void **)&graph->lnpwgts);
  GKfree((void **)&graph->gnpwgts);
  GKfree((void **)&graph->peind);
  GKfree((void **)&graph->sendptr);
  GKfree((void **)&graph->sendind);
  GKfree((void **)&graph->recvptr);
  GKfree((void **)&graph->recvind);
  GKfree((void **)&graph->imap);
  GKfree((void **)&graph->rlens);
  GKfree((void **)&graph->slens);
  GKfree((void **)&graph->rcand);
  GKfree((void **)&graph->pexadj);
  GKfree((void **)&graph->peadjncy);
  GKfree((void **)&graph->peadjloc);

  free(graph);
}



/*************************************************************************
* This function deallocates any memory stored in a graph
**************************************************************************/
void FreeInitialGraphAndRemap(GraphType *graph, int wgtflag, int freevsize) 
{
  int i, nedges;
  idxtype *adjncy, *imap;

  nedges = graph->nedges;
  adjncy = graph->adjncy;
  imap   = graph->imap;

  if (imap != NULL) {
    for (i=0; i<nedges; i++)
      adjncy[i] = imap[adjncy[i]];  /* Apply local to global transformation */
  }

  /* Free Metis's things */
  GKfree((void **)&graph->match);
  GKfree((void **)&graph->cmap);
  GKfree((void **)&graph->lperm);
  GKfree((void **)&graph->where);
  GKfree((void **)&graph->label);
  GKfree((void **)&graph->rinfo);
  GKfree((void **)&graph->nrinfo);
  GKfree((void **)&graph->nvwgt);
  GKfree((void **)&graph->lpwgts);
  GKfree((void **)&graph->gpwgts);
  GKfree((void **)&graph->lnpwgts);
  GKfree((void **)&graph->gnpwgts);
  GKfree((void **)&graph->sepind);
  GKfree((void **)&graph->peind);
  GKfree((void **)&graph->sendptr);
  GKfree((void **)&graph->sendind);
  GKfree((void **)&graph->recvptr);
  GKfree((void **)&graph->recvind);
  GKfree((void **)&graph->imap);
  GKfree((void **)&graph->rlens);
  GKfree((void **)&graph->slens);
  GKfree((void **)&graph->rcand);
  GKfree((void **)&graph->pexadj);
  GKfree((void **)&graph->peadjncy);
  GKfree((void **)&graph->peadjloc);

  if (freevsize)
    GKfree((void **)&graph->vsize);
  if ((wgtflag&2) == 0) 
    GKfree((void **)&graph->vwgt);
  if ((wgtflag&1) == 0) 
    GKfree((void **)&graph->adjwgt);

  free(graph);
}
