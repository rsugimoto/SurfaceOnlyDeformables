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




// Programmer: Sung-Eui Yoon (Dec-26, 2004)
#ifndef _Layout_Metrics_
#define _Layout_Metrics_


/*
Usage:

	CMetrics COMetric;

	COMetric.InitOrdering ();
	
	COMetric.AddOrdering ();	// 1st ordering	
	COMetric.AddOrdering ();	// 2nd ordering	

	int Result = COMetric.GetBestOrdering ();	// return 1 or 2 in this case


	// reuse this class call InitOrdering and continue same job.	
	COMetric.InitOrdering ();

	COMetric.AddOrdering ();	// 1st ordering	
	.....

*/


#include <map>
using namespace std;

namespace OpenCCL 
{

// ---------------------------------------------------------------------
struct ltstr_map
{
	bool operator () (const unsigned int a, const unsigned int b) const
	{
		return a < b; 
	}
};

// CHyperPlane is a sorted as edge length. Each edge length has its counts of occurrence.
typedef map <const unsigned int, int, ltstr_map> CHyperPlane;
// ------------------------------------------------------------




class CMetrics
{
	bool m_bMetricType;		// true --> cache-oblivious
					// false -> cache-aware

	CHyperPlane m_hpBestOrdering;	// Best or initial ordering
	int m_NumAddedOrdering;
	int m_BestOrdering;

	float PerformCOMetric (CHyperPlane & HyperPlane);	// cache-oblvious metric
	void InitOrdering (int NumEdgeLengths, int * EdgeLengths);

public:
	CMetrics (void);
	void InitOrdering (void);	// clean records of AddOrdering	
	int AddOrdering (int NumEdgeLengths, int * EdgeLengths);
	int GetBestOrdering (void);	// return the best ordering among added orderings.



	
};



}	// end of namespace
#endif


