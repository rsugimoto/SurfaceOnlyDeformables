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



#ifndef _VDT_ACTIVE_LIST_
#define _VDT_ACTIVE_LIST_


namespace OpenCCL
{

#define TRUE	1
#define FALSE 	0


// this node can be used as capsulation for arbitrary class and 
// be used for CActiveList

// this is not generic template
template <class ptrDataType>
class CActiveNode
{
public :
	CActiveNode * m_pNext, * m_pPrev;	

	//this contain pointer
	ptrDataType m_Data;

	CActiveNode (void)
	{
		m_pNext = NULL;
		m_pPrev = NULL;
	}
};

// ptrDataType should be a pointer.
template <class ptrDataType>
class CActiveList  
{
public :
        ptrDataType m_pStart, m_pEnd;
	int m_Size;

	// iterator pointer
	ptrDataType m_pCurrentNode;

	// Mark
	ptrDataType m_pMark;			// store one node for user-defined purpose.

        inline CActiveList (void);
        inline ~CActiveList (void);
	inline void InitList (ptrDataType pStart, ptrDataType pEnd);
	inline int Delete (ptrDataType pNode);
    inline void Clear(bool deleteNodes = false);
	inline void Add (ptrDataType pNode);
	inline void ForceAdd (ptrDataType pNode);
	inline void AddNext (ptrDataType pPivotNode, ptrDataType pNode);
	inline void AddBefore (ptrDataType pPivotNode, ptrDataType pNode);
	inline void AddatEnd (ptrDataType pNode);
	inline ptrDataType Head (void);
    inline ptrDataType End (void);
	inline int IsEmpty (void);
	inline int Size (void);
	inline bool IsValid (ptrDataType pNode);

	// operation with Mark
	inline int DeleteWithMark (ptrDataType pNode);
	inline void SetMark (ptrDataType pNode);
	inline ptrDataType GetMark (void);

	// simple iterator.
	inline void InitIteration (void);
	inline void InitIteration (ptrDataType pNode);
	inline void SetCurrent (ptrDataType pNode);
	inline int IsEnd (void);
	inline ptrDataType GetCurrent (void);
	inline void Advance (void);
	inline void BackAdvance (void);

};


template <class ptrDataType>
inline CActiveList <ptrDataType>::CActiveList (void)
{
	m_pStart = m_pEnd = NULL;
	//InitList ();

	/*
	m_pStart = new ptrDataType;
	m_pEnd = new ptrDataType;

	InitList (m_pStart, m_pEnd);
	*/
}

template <class ptrDataType>
inline CActiveList <ptrDataType>::~CActiveList (void)
{
	if (m_pStart != NULL) {
		delete m_pStart;
		m_pStart = NULL;
	}
	if (m_pEnd != NULL) {
		delete m_pEnd;
		m_pEnd = NULL;
	}
}
// to use this class, we have create Start and End node. then assign the pointers of this
// to this function.
template <class ptrDataType>
inline void CActiveList <ptrDataType>::InitList (ptrDataType pStart, ptrDataType pEnd)
{
	m_pStart = pStart;
	m_pEnd = pEnd;

	// make a double list.
	m_pStart->m_pPrev = NULL;
	m_pStart->m_pNext = m_pEnd;

	m_pEnd->m_pPrev = m_pStart;
	m_pEnd->m_pNext = NULL;

	m_Size = 0;
	m_pCurrentNode = 0;
	m_pMark = 0;
}

template <class ptrDataType>
inline int CActiveList <ptrDataType>::IsEmpty (void)
{
	if (m_pStart->m_pNext == m_pEnd)
		return TRUE;

	return FALSE;	

}

template <class ptrDataType>
inline bool CActiveList <ptrDataType>::IsValid (ptrDataType pNode)
{
	if (pNode->m_pNext == NULL || pNode->m_pPrev == NULL)
		return false;

	return true;
}

template <class ptrDataType>
inline int CActiveList <ptrDataType>::Delete (ptrDataType pNode)
{
	if (pNode->m_pNext == NULL || pNode->m_pPrev == NULL)  // if this isn't an active one. 
	       return FALSE;	

	if (pNode == m_pCurrentNode)
		SetCurrent (m_pCurrentNode->m_pPrev);

	pNode->m_pPrev->m_pNext = pNode->m_pNext;
	pNode->m_pNext->m_pPrev = pNode->m_pPrev;

	pNode->m_pPrev = NULL;
	pNode->m_pNext = NULL;

	m_Size--;
	return TRUE;
}

template <class ptrDataType>
inline void CActiveList <ptrDataType>::Clear (bool deleteNodes)
{
    m_Size = 0;
    if (deleteNodes)
    {
        ptrDataType cur, next;
        cur = m_pStart->m_pNext;
        while (0 != cur && m_pEnd != cur)
        {
            next = cur->m_pNext;
            delete cur;
            cur = next;
        }
    }
    m_pEnd->m_pPrev = m_pStart;
    m_pStart->m_pNext = m_pEnd;
}

// if Mark is deleted, changed Mark into next one.
template <class ptrDataType>
inline int CActiveList <ptrDataType>::DeleteWithMark (ptrDataType pNode)
{
	if (pNode->m_pNext == NULL || pNode->m_pPrev == NULL) // temporary solution.
	       return FALSE;	

	if (pNode == m_pMark) {		// user-purpose. 
		m_pMark = m_pMark->m_pNext;
	}

	pNode->m_pPrev->m_pNext = pNode->m_pNext;
	pNode->m_pNext->m_pPrev = pNode->m_pPrev;

	pNode->m_pPrev = NULL;
	pNode->m_pNext = NULL;

	m_Size--;

	return TRUE;		// it means it delete element.
}

template <class ptrDataType>
inline void CActiveList <ptrDataType>::SetMark (ptrDataType pNode)
{
	m_pMark = pNode;
}

template <class ptrDataType>
inline ptrDataType CActiveList <ptrDataType>::GetMark (void)
{
	return m_pMark;
}


template <class ptrDataType>
inline void CActiveList <ptrDataType>::Add (ptrDataType pNode)
{
	if (pNode->m_pNext != NULL)	// already inserted in list
		return;

	// add node after m_Start, which is a root node
        pNode->m_pNext = m_pStart->m_pNext;
        pNode->m_pPrev = m_pStart;

        pNode->m_pNext->m_pPrev = pNode;
       	m_pStart->m_pNext = pNode;

	m_Size++;
}
template <class ptrDataType>
inline void CActiveList <ptrDataType>::ForceAdd (ptrDataType pNode)
{
	if (pNode->m_pNext != NULL) {	// already inserted in list
		Delete (pNode);
	}

	// add node after m_Start, which is a root node
        pNode->m_pNext = m_pStart->m_pNext;
        pNode->m_pPrev = m_pStart;

        pNode->m_pNext->m_pPrev = pNode;
       	m_pStart->m_pNext = pNode;

	m_Size++;
}
template <class ptrDataType>
inline void CActiveList <ptrDataType>::AddatEnd (ptrDataType pNode)
{
	if (pNode->m_pNext != NULL)	// already inserted in list
		return;

	// add node before m_pEnd, which is a root node
	
	pNode->m_pNext = m_pEnd;
	pNode->m_pPrev = m_pEnd->m_pPrev;

	m_pEnd->m_pPrev->m_pNext = pNode;
       	m_pEnd->m_pPrev = pNode;

	m_Size++;
}

template <class ptrDataType>
inline void CActiveList <ptrDataType>::AddNext (ptrDataType pPivotNode, ptrDataType pNode)
{
	if (pNode->m_pNext != NULL) {	// already inserted in list
	//	printf ("To check if it might be unnecessary code.\n");
	//	exit (-1);
		return;
	}

	// add node after m_pPivotNode, which is a root node
        pNode->m_pNext = pPivotNode->m_pNext;
        pNode->m_pPrev = pPivotNode;

        pNode->m_pNext->m_pPrev = pNode;
       	pPivotNode->m_pNext = pNode;

	m_Size++;
}

template <class ptrDataType>
inline void CActiveList <ptrDataType>::AddBefore (ptrDataType pPivotNode, ptrDataType pNode)
{
	if (pNode->m_pNext != NULL) {	// already inserted in list
	//	printf ("To check if it might be unnecessary code.\n");
	//	exit (-1);
		return;
	}

	// add node before m_pPivotNode
	// 
        pNode->m_pNext = pPivotNode;
        pNode->m_pPrev = pPivotNode->m_pPrev;

	pPivotNode->m_pPrev->m_pNext = pNode;
	pPivotNode->m_pPrev = pNode;

	m_Size++;
}


template <class ptrDataType>
inline ptrDataType CActiveList <ptrDataType>::Head (void)
{
	return m_pStart->m_pNext;
}

template <class ptrDataType>
inline int CActiveList <ptrDataType>::Size (void)
{
	return m_Size;
}
template <class ptrDataType>
inline void CActiveList <ptrDataType>::InitIteration (void)
{
	ptrDataType pRootNode = Head ();

	//SetCurrent (pRootNode);

	// above code produce message if list is empty.	
	m_pCurrentNode = pRootNode;

}

template <class ptrDataType>
inline void CActiveList <ptrDataType>::InitIteration (ptrDataType pNode)
{
	ptrDataType pRootNode = pNode;

	//SetCurrent (pRootNode);

	// above code produce message if list is empty.	
	m_pCurrentNode = pRootNode;

}


template <class ptrDataType>
inline void CActiveList <ptrDataType>::SetCurrent (ptrDataType pNode)
{
	#ifdef DEBUG_MODE
	if (pNode->m_pNext == NULL) {
		printf ("Error : Invalid Current Node\n");
		exit (-1);
	}
	#endif

	m_pCurrentNode = pNode;
}

template <class ptrDataType>
inline int CActiveList <ptrDataType>::IsEnd (void)
{
	if (m_pCurrentNode == m_pEnd)
		return TRUE;

	return FALSE;
}
template <class ptrDataType>
inline ptrDataType CActiveList <ptrDataType>::GetCurrent (void)
{
	return m_pCurrentNode;	
}
template <class ptrDataType>
inline void CActiveList <ptrDataType>::Advance (void)
{
	m_pCurrentNode = m_pCurrentNode->m_pNext;	
}
template <class ptrDataType>
inline void CActiveList <ptrDataType>::BackAdvance (void)
{
	m_pCurrentNode = m_pCurrentNode->m_pPrev;	
}


}	// end of namespace
#endif
