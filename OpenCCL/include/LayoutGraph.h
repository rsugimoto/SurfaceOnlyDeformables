/******************************************************************************\

Copyright (c) <2015>, <UNC-CH>
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


---------------------------------
|Please send all BUG REPORTS to:  |
|                                 |
|   sungeui@cs.kaist.ac.kr        |
|   sungeui@gmail.com             |
|                                 |
---------------------------------


The authors may be contacted via:

Mail:         Sung-Eui Yoon
              Dept. of Computer Science, E3-1
              KAIST
              291 Daehak-ro(373-1 Guseong-dong), Yuseong-gu
              DaeJeon, 305-701
              Republic of Korea

\*****************************************************************************/




#ifndef _LAYOUT_GRAPH_H
#define _LAYOUT_GRAPH_H


#include "OpenCCL.h"







#include "VArray.h"

#ifdef WIN32

#ifdef USE_STL_Hashmap
#include <hash_map>
#else
#include "_hash_map.h"
#endif

#else	// linux

#ifdef USE_STL_Hashmap
#include <ext/hash_map>
using __gnu_cxx::hash_map; 
using __gnu_cxx::hash; 
#else
#include "_hash_map.h"
#endif


#endif


#include "VDTActiveList.h"
#include <queue>



using std::queue;

namespace OpenCCL 
{


static const int g_NumSubdivision = 4;



#ifdef USE_STL_Hashmap
// hashmap to detect unique vertex
struct eqNeighbor
{
        bool operator()(int p1, int p2) const
        {
                if (p1 ==  p2)
                        return 1;
                return 0;
        }
};
typedef hash_map <int, int, hash <int>, eqNeighbor> CIntHashMap;
#else

// I got this code from 
// http://www.csua.berkeley.edu/~emin/source_code/hash_map/

struct SimpleIDHasher : public MyHasherModel<int> {
public:
  // Since we are hashing based on integers we can just use the
  // GENERIC_HASH macro.  GENERIC_HASH is a macro which takes
  // an integer and scrambles it up.
  int operator()(const int & key)const {return GENERIC_HASH(key);}

};

struct IDEqualCmp {
  bool operator()(const int x, const int y) const {return x == y;}
};


typedef my_hash_map <int, int, SimpleIDHasher, IDEqualCmp> CIntHashMap;




#endif


#define NULL_IDX		-1
class CLayoutVertex
{
public:
	//int m_v;		// indicate a vertex
	//int m_c;		// indicate an edge inciden on the vertex


	int m_ID;			// a cluster id
	int m_Order;		// current Order in the list
	int m_VWeight;
	CIntHashMap m_Neighbor;	// neighbor units


	VArray <int> m_Vertices;	// not for leaf units, holding leaf units duing partitioing
	CLayoutVertex * m_pNext, * m_pPrev;
	

	CLayoutVertex * m_pChild [g_NumSubdivision], * m_pParent;	
	int m_Depth;
	char m_NumChild;
	bool m_bRefined;



	CLayoutVertex (void);
	bool AddNeighbor (int Neighbor, int Weight);
	bool InsertNeighbor (int Neighbor);


};

class CLayoutEdge
{
public:
	int m_v [2];		// two vertex
//	int m_c [2];		// indicate next edge incident on m_v
};

class CLayoutGraph 
{


	CLayoutVertex * m_Vertex;
	CLayoutEdge * m_Edge;

	VArray <CLayoutEdge> m_VArrayEdge;

	int m_NumVertex;
	int m_NumEdge;




	// for partition
	CLayoutVertex * m_Unit;
	int m_NumUnit;
	int * m_MapOID2CID;		// map between original Tri id and node id
	CLayoutVertex * m_pRootUnit;
	CActiveList <CLayoutVertex *> m_VList;
	// ----------------------------


	// for partition
	int GetOrder (int v);
	void ComputeOrdering (CLayoutVertex * pNode, queue <CLayoutVertex *> & Queue);
	void ConstructNeighbor (CLayoutVertex * pRoot);
	int PostprocessingMetis (int SrcNumCluster, int DestNumCluster, int AvgClusterSize);
	bool GetSubComponent (VArray <int> & Cluster, VArray <VArray <int> > & Components);
	bool AssignSubComponent (VArray <int> & ParentSubCom, 
		VArray <int> & SubCom, VArray <VArray <int> > & Clusters, 
		int AvgClusterSize, bool ClusterGranul);
	void PartitionUnits (CLayoutVertex * pNode, queue <CLayoutVertex *> & Queue);
	void GetCombination (int NumUnit, VArray <VArray <int> > & Combination, int Which = 0);
	float CostFunc (int IdxGap, bool ClusterLevel);
	float GetCost (CLayoutVertex * pPivot, int DestUnit);
	bool IsOut (int SrcID, CLayoutVertex ** pChild, int NumChild);
	float GetCost (VArray <int> & Order, CLayoutVertex * pParent, 
		bool & Jump);

	public:
	CLayoutGraph (int NumVertex);
	~CLayoutGraph (void);

	void AddEdge (int v1, int v2);
	void ComputeOrdering (int * OrderArray);


};


}	// end of namespace
#endif
