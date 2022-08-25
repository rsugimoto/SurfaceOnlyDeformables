#ifndef METIS_H
#define METIS_H

typedef int idxtype;

extern "C" {

void METIS_PartMeshNodal(int *, int *, idxtype *, int *, int *, int *, int *, idxtype *, idxtype *);

void METIS_PartMeshDual(int *, int *, idxtype *, int *, int *, int *, int *, idxtype *, idxtype *);

void METIS_PartGraphRecursive(int*, idxtype*, idxtype*, idxtype*, idxtype*, int*, int*, int*, int*, int*, idxtype*);

void METIS_PartGraphKway(int*, idxtype*, idxtype*, idxtype*, idxtype*, int*, int*, int*, int*, int*, idxtype*);

void METIS_PartGraphVKway(int*, idxtype*, idxtype*, idxtype*, idxtype*, int*, int*, int*, int*, int*, idxtype*);

};

#endif

