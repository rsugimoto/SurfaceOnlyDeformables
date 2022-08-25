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





//-----------------------------------------------------------------------------
// @ VArray.h
// ---------------------------------------------------------------------------
// Template class for a dynamic array.
//-----------------------------------------------------------------------------


#ifndef __VARRAY_H__
#define __VARRAY_H__

//-----------------------------------------------------------------------------
//-- Includes ----------------------------------------------------------------
//-----------------------------------------------------------------------------

#include <assert.h>
#include <stdio.h>
#include <memory.h>
//#include <binstream.h>
//#include <iostream.h>


//-----------------------------------------------------------------------------
//-- Forward Declarations ----------------------------------------------------
//-----------------------------------------------------------------------------
template <class Type> class VArray ;

/*
template <class Type>
inline istream& Read(istream& in, VArray<Type>& arr);
template <class Type>
inline ostream& Write(ostream& out, const VArray<Type>& arr);
*/


//-----------------------------------------------------------------------------
//-- Typedefs, Structs, Classes ----------------------------------------------
//-----------------------------------------------------------------------------

template <class Type>
class VArray 
{
  typedef int (* CompareFunc)(const Type &, const Type &);
  public:
    // Constructors/Destructor
    inline VArray();
    inline VArray(int size);
    inline VArray(const VArray &arr);
    inline ~VArray();

    // element-wise assignment. Needs Type::operator=() 
    inline VArray& operator=(const VArray& arr);
    // Equality test for entire array. 
    // Requires Type::operator==() for comparison.
    inline bool operator==(const VArray& arr) const;
    // Gets the number of valid elements in array
    inline int GetCount() const;
    // Gets the number of valid elements in array
    inline int Size() const;
    // Sets the number of valid elements in array. Can shrink/grow.
    inline void SetCount(int newCount);
    // Gets allocated capacity of the array.
    inline int GetCapacity() const;
    // Sets allocated capacity of the array. Can shrink/grow.
    void SetCapacity(int newCapacity);
    // Clears the VArray
    inline void Clear(bool shrink=true);
    // Append an element. Might grow capacity.
    inline void Append();
    // Append an element. Might grow capacity.
    inline void Append(const Type& data);
    // Append an array. Might grow capacity.
    inline void Append(const VArray& data);
    // Append an element uniquely.
    // Requires Type::operator==() for comparison.
    inline bool AppendUnique(const Type& data);
    // Append an array, but only unique elements. 
    // Requires Type::operator==() for comparison.
    inline void AppendUnique(const VArray& data);
    // Remove index'th element.
    inline void Remove(int index);
    // Make Capacity == Count
    inline void Pack();
    // Grow count by a given amount. -1 means double.
    inline void Grow(int amt=-1);
    // Grow capacity by a given amount. -1 means double.
    inline void GrowCapacity(int amt=-1);
    // array access.
    inline const Type& operator [](const int i) const;
    inline Type& operator[](const int i);
    // array access by way of function call. same as [i]
    inline const Type& Get(const int i) const;
    inline Type& Get(const int i);
    // First element in array.
    inline const Type& Head() const;
    inline Type& Head();
    // Last element in array.
    inline const Type& Tail() const;
    inline Type& Tail();
    // return the actual array storage.
    inline Type* GetArray();
    inline const Type* GetArray() const;
    // Detatch the array storage w/o freeing it for use externally.
    inline Type* DetachArray();
    // Discard the old array and use this one. Resets capacity and count.
    // Takes ownership of array, so will free it later.
    inline void SetArray(Type *arr, int size);
    // Linear search for an element.
    // Requires Type::operator==() for comparison.
    inline int IndexOf(const Type& data) const;
    //friend istream& TEMPL_FRIEND(Read)(istream& in, VArray& arr);
    //friend ostream& TEMPL_FRIEND(Write)(ostream& out, const VArray& arr);
    
    // QuickSort the array
    // Requires Type::operator>() for comparison.
    void QuickSort();
    void QuickSort(CompareFunc compare);
    // QuickSort an array of ptrs.
    // Requires (*Type)::operator>() for comparison.
    void QuickSortPtr();
    void QuickSortPtr(CompareFunc compare);
    // QuickSort a subarray [p..r]
    // Requires Type::operator>() for comparison.
    void QuickSort(int p, int r);
    void QuickSort(int p, int r, CompareFunc compare);
    // QuickSortPtr a subarray [p..r]
    // Requires (*Type)::operator>() for comparison.
    void QuickSortPtr(int p, int r);
    void QuickSortPtr(int p, int r, CompareFunc compare);
  protected:
    // Real workhorse of QuickSort.
    // Requires Type::operator>() for comparison.
    int Partition(int p, int r);
    int Partition(int p, int r, CompareFunc compare);
    // Real workhorse of QuickSortPtr.
    // Requires (*Type)::operator>() for comparison.
    int PartitionPtr(int p, int r);
    int PartitionPtr(int p, int r, CompareFunc compare);

    int count;
    int capacity;
    Type* array;
  private:
};

//-----------------------------------------------------------------------------
//-- Function Definitions ----------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// @ VArray::VArray()
// ---------------------------------------------------------------------------
// Constructors
//-----------------------------------------------------------------------------
template <class Type>
inline VArray<Type>::VArray()
{
  count = 0;
  capacity = 0;
  array = NULL;
}

template <class Type>
inline VArray<Type>::VArray(int size)
{
  count = 0;
  capacity = 0;
	array = NULL;
  SetCount(size);
}

template <class Type>
inline VArray<Type>::VArray(const VArray<Type> &arr) 
{
  count = 0;
  capacity = 0;
	array = NULL;
  *this = arr;
}

//-----------------------------------------------------------------------------
// @ VArray::~VArray()
// ---------------------------------------------------------------------------
// Destructor
//-----------------------------------------------------------------------------
template <class Type>
inline VArray<Type>::~VArray() 
{ 
  if (array) {
    delete[] array; 
    array = NULL; 
  }
}

//-----------------------------------------------------------------------------
// @ VArray::operator=()
// ---------------------------------------------------------------------------
// element-wise assignment. Needs Type::operator=() 
//-----------------------------------------------------------------------------
template <class Type>
inline VArray<Type>& VArray<Type>::operator=(const VArray<Type>& arr)
{
  if (&arr==this) return *this;

  SetCount(arr.count);
  for (int i=0;i<count;i++) {
    array[i] = arr.array[i];
  }
  return *this;
}

//-----------------------------------------------------------------------------
// @ VArray::operator==()
// ---------------------------------------------------------------------------
// Equality test for entire array. 
// Requires Type::operator==() for comparison.
//-----------------------------------------------------------------------------
template <class Type>
inline bool VArray<Type>::operator==(const VArray<Type>& arr) const
{
  if (count != arr.count) return false;

  for (int i=0;i<count;i++) {
    if (!(array[i] == arr.array[i])) {
      return false;
    }
  }
  return true;
}

//-----------------------------------------------------------------------------
// @ VArray::GetCount()
// ---------------------------------------------------------------------------
// Gets the number of valid elements in array
//-----------------------------------------------------------------------------
template <class Type>
inline int VArray<Type>::GetCount() const
{ 
  return count; 
}

//-----------------------------------------------------------------------------
// @ VArray::Size()
// ---------------------------------------------------------------------------
// Gets the number of valid elements in array
//-----------------------------------------------------------------------------
template <class Type>
inline int VArray<Type>::Size() const
{ 
  return count; 
}

//-----------------------------------------------------------------------------
// @ VArray::SetCount()
// ---------------------------------------------------------------------------
// Sets the number of valid elements in array. Can shrink/grow.
//-----------------------------------------------------------------------------
template <class Type>
inline void VArray<Type>::SetCount(int newCount) 
{
  if (newCount == count) return;

  if (newCount > capacity) {
    SetCapacity(newCount);
    count = capacity;
  } else {
    count = newCount;
  }
}

//-----------------------------------------------------------------------------
// @ VArray::GetCapacity()
// ---------------------------------------------------------------------------
// Gets allocated capacity of the array.
//-----------------------------------------------------------------------------
template <class Type>
inline int VArray<Type>::GetCapacity() const
{
  return capacity;
}

//-----------------------------------------------------------------------------
// @ VArray::SetCapacity()
// ---------------------------------------------------------------------------
// Sets allocated capacity of the array. Can shrink/grow.
//-----------------------------------------------------------------------------
template <class Type>
void VArray<Type>::SetCapacity(int newCapacity)
{
  if (newCapacity == 0) {
    if (array) delete[] array;
    array = NULL;
    capacity=0;
    count=0;
  } else if (newCapacity != capacity) {
    Type* newArr = new Type[newCapacity];
    assert(newArr != NULL);
    capacity=newCapacity;
    if (capacity < count) count = capacity;
    for (int i=0;i<count;i++) {
      newArr[i] = array[i];
    }
    if (array) delete[] array;
    array = newArr;
  }
}

//-----------------------------------------------------------------------------
// @ VArray::Clear()
// ---------------------------------------------------------------------------
// Clears the VArray
//-----------------------------------------------------------------------------
template <class Type>
inline void VArray<Type>::Clear(bool shrink)
{ 
  if (shrink) {
    SetCapacity(0);
  } else {
    SetCount(0);
  }
}

//-----------------------------------------------------------------------------
// @ VArray::Append()
// ---------------------------------------------------------------------------
// Append an element. Might grow capacity.
//-----------------------------------------------------------------------------
template <class Type>
inline void VArray<Type>::Append(const Type& data) 
{
  Append();
  array[count-1] = data;
}

template <class Type>
inline void VArray<Type>::Append() 
{
  if (count == capacity) {
    SetCapacity(capacity==0?4:capacity*2);
  }
  count++;
}

//-----------------------------------------------------------------------------
// @ VArray::Append()
// ---------------------------------------------------------------------------
// Append an array. Might grow capacity.
//-----------------------------------------------------------------------------
template <class Type>
inline void VArray<Type>::Append(const VArray<Type>& data)
{
  if (data.count == 0) return; // nothing to do

  // First see if doubling the array size is sufficient.
  // This avoids O(n^2) behavior when repeatedly Appending lots 
  // of small arrays.
  if (count+data.count > capacity) {
    if (count+data.count <= capacity*2) {
      SetCapacity(capacity==0?4:capacity*2);
    }
    // If that's not enough room just set the capacity to what we need.
    else {
      SetCapacity(count+data.count);
    }
  }

  // Append the new data
  for (int i=0;i<data.count;i++) {
    array[count++] = data.array[i];
  }
}

//-----------------------------------------------------------------------------
// @ VArray::AppendUnique()
// ---------------------------------------------------------------------------
// Append an element uniquely.
// Requires Type::operator==() for comparison.
//-----------------------------------------------------------------------------
template <class Type>
inline bool VArray<Type>::AppendUnique(const Type& data)
{
  if (IndexOf(data) == -1) {
    Append(data);
    return true;
  }
  return false;
}

//-----------------------------------------------------------------------------
// @ VArray::AppendUnique()
// ---------------------------------------------------------------------------
// Append an array, but only unique elements. 
// Requires Type::operator==() for comparison.
//-----------------------------------------------------------------------------
template <class Type>
inline void VArray<Type>::AppendUnique(const VArray<Type>& data)
{
  for (int i=0;i<data.count;i++) {
    AppendUnique(data.array[i]);
  }
}

//-----------------------------------------------------------------------------
// @ VArray::Remove()
// ---------------------------------------------------------------------------
// Remove index'th element.
//-----------------------------------------------------------------------------
template <class Type>
inline void VArray<Type>::Remove(int index)
{
	assert(index<count && index >= 0); 
  for (int i=index;i<count-1;i++) {
	  array[i] = array[i+1];
  }
  count--;
}
    
//-----------------------------------------------------------------------------
// @ VArray::Pack()
// ---------------------------------------------------------------------------
// Make Capacity == Count
//-----------------------------------------------------------------------------
template <class Type>
inline void VArray<Type>::Pack() 
{
  SetCapacity(count);
}

//-----------------------------------------------------------------------------
// @ VArray::Grow()
// ---------------------------------------------------------------------------
// Grow count by a given amount. -1 means double.
//-----------------------------------------------------------------------------
template <class Type>
inline void VArray<Type>::Grow(int amt)
{
  GrowCapacity(count+amt-capacity);
  count = capacity;
}

//-----------------------------------------------------------------------------
// @ VArray::GrowCapacity()
// ---------------------------------------------------------------------------
// Grow capacity by a given amount. -1 means double.
//-----------------------------------------------------------------------------
template <class Type>
inline void VArray<Type>::GrowCapacity(int amt)
{
  if (amt==0) return;
  if (amt==-1) {
    SetCapacity(capacity==0?4:capacity*2);
  } else {
    SetCapacity(capacity+amt);
  }
}

//-----------------------------------------------------------------------------
// @ VArray::operator[]()
// ---------------------------------------------------------------------------
// array access.
//-----------------------------------------------------------------------------
template <class Type>
inline const Type& VArray<Type>::operator [](const int i) const 
{ 
  assert(array && i<count && i >= 0);
  return array[i];
}

template <class Type>
inline Type& VArray<Type>::operator[](const int i)
{ 
  assert(array && i<count && i >= 0);
  return array[i];
}

//-----------------------------------------------------------------------------
// @ VArray::Get()
// ---------------------------------------------------------------------------
// array access by way of function call. same as [i]
//-----------------------------------------------------------------------------
template <class Type>
inline const Type& VArray<Type>::Get(const int i) const
{
  return (*this)[i];
}

template <class Type>
inline Type& VArray<Type>::Get(const int i) 
{
  return (*this)[i];
}

//-----------------------------------------------------------------------------
// @ VArray::Head()
// ---------------------------------------------------------------------------
// First element in array.
//-----------------------------------------------------------------------------
template <class Type>
inline const Type& VArray<Type>::Head() const
{
  assert(count>0);
  return array[0];
}

template <class Type>
inline Type& VArray<Type>::Head()
{
  assert(count>0);
  return array[0];
}

//-----------------------------------------------------------------------------
// @ VArray::Tail()
// ---------------------------------------------------------------------------
// Last element in array.
//-----------------------------------------------------------------------------
template <class Type>
inline const Type& VArray<Type>::Tail() const
{
  assert(count>0);
  return array[count-1];
}

template <class Type>
inline Type& VArray<Type>::Tail()
{
  assert(count>0);
  return array[count-1];
}

//-----------------------------------------------------------------------------
// @ VArray::GetArray()
// ---------------------------------------------------------------------------
// return the actual array storage.
//-----------------------------------------------------------------------------
template <class Type>
inline Type* VArray<Type>::GetArray() 
{
  return array;
}

template <class Type>
inline const Type* VArray<Type>::GetArray() const
{
  return array;
}

//-----------------------------------------------------------------------------
// @ VArray::DetachArray()
// ---------------------------------------------------------------------------
// Detatch the array storage w/o freeing it for use externally.
//-----------------------------------------------------------------------------
template <class Type>
inline Type* VArray<Type>::DetachArray()
{
  Type *res = array;
  array = NULL;
  capacity = 0;
  count = 0;
  return res;
}

//-----------------------------------------------------------------------------
// @ VArray::SetArray()
// ---------------------------------------------------------------------------
// Discard the old array and use this one. Resets capacity and count.
// Takes ownership of array, so will free it later.
//-----------------------------------------------------------------------------
template <class Type>
inline void VArray<Type>::SetArray(Type *arr, int size)
{
  Clear();
  array = arr;
  capacity = count = size;
}

//-----------------------------------------------------------------------------
// @ VArray::IndexOf()
// ---------------------------------------------------------------------------
// Linear search for an element.
// Requires Type::operator==() for comparison.
//-----------------------------------------------------------------------------
template <class Type>
inline int VArray<Type>::IndexOf(const Type& data) const
{
  for (int i=0;i<count;i++) {
    if (data==array[i]) return i;
  }
  return -1;
}

/*
template <class Type>
inline istream& Read(istream& in, VArray<Type>& arr)
{
  int count;
  Read(in, count);
  arr.SetCount(count);
  for (int i=0;i<count;i++) {
    Read(in,arr.array[i]);
  }
  return in;
}

template <class Type>
inline ostream& Write(ostream& out, const VArray<Type>& arr)
{
  Write(out, arr.count);
  for (int i=0;i<arr.count;i++) {
    Write(out,arr.array[i]);
  }
  return out;
}
*/

//-----------------------------------------------------------------------------
// @ VArray::QuickSort()
// ---------------------------------------------------------------------------
// Quicksort the VArray.
// Requires Type::operator>() for comparison.
//-----------------------------------------------------------------------------
template <class Type>
void VArray<Type>::QuickSort()
{
  QuickSort(0,count-1);
}

template <class Type>
void VArray<Type>::QuickSort(CompareFunc compare)
{
  QuickSort(0,count-1, compare);
}

//-----------------------------------------------------------------------------
// @ VArray::QuickSort()
// ---------------------------------------------------------------------------
// Quicksort the VArray.
// Requires Type::operator>() for comparison.
//-----------------------------------------------------------------------------
template <class Type>
void VArray<Type>::QuickSort(int p, int r)
{
  if (count == 0) return;
  int q;
  
  while (p < r) {
    q = Partition(p,r);
    QuickSort(p,q);
    p = q+1;
  }
}

template <class Type>
void VArray<Type>::QuickSort(int p, int r, CompareFunc compare)
{
  if (count == 0) return;
  int q;
  
  while (p < r) {
    q = Partition(p,r,compare);
    QuickSort(p,q,compare);
    p = q+1;
  }
}

//-----------------------------------------------------------------------------
// @ VArray::Partition()
// ---------------------------------------------------------------------------
// Real workhorse of QuickSort()....
// Requires Type::operator>() for comparison.
//-----------------------------------------------------------------------------
template <class Type>
int VArray<Type>::Partition(int p, int r)
{
  Type t;
  Type x = array[p];
  int i = p-1;
  int j = r+1;
  
  while(1) {
    do {
      j--;
    } while (array[j] > x);
    do {
      i++;
    } while (x > array[i]);
    if (i < j) {
      t = array[i];
      array[i] = array[j];
      array[j] = t;
    } else {
      return j;
    }
  }
}

template <class Type>
int VArray<Type>::Partition(int p, int r, CompareFunc compare)
{
  Type t;
  Type x = array[p];
  int i = p-1;
  int j = r+1;
  
  while(1) {
    do {
      j--;
    } while (compare(array[j],x) > 0);
    do {
      i++;
    } while (compare(x,array[i]) > 0);
    if (i < j) {
      t = array[i];
      array[i] = array[j];
      array[j] = t;
    } else {
      return j;
    }
  }
}

//-----------------------------------------------------------------------------
// @ VArray::QuickSortPtr()
// ---------------------------------------------------------------------------
// QuickSort an VArray of ptrs.
// Requires (*Type)::operator>() for comparison.
//-----------------------------------------------------------------------------
template <class Type>
void VArray<Type>::QuickSortPtr()
{
  QuickSortPtr(0,count-1);
}

template <class Type>
void VArray<Type>::QuickSortPtr(CompareFunc compare)
{
  QuickSortPtr(0,count-1, compare);
}

//-----------------------------------------------------------------------------
// @ VArray::QuickSortPtr()
// ---------------------------------------------------------------------------
// QuickSortPtr a subVArray [p..r]
// Requires (*Type)::operator>() for comparison.
//-----------------------------------------------------------------------------
template <class Type>
void VArray<Type>::QuickSortPtr(int p, int r)
{
  if (count == 0) return;
  int q;
  
  while (p < r) {
    q = PartitionPtr(p,r);
    QuickSortPtr(p,q);
    p = q+1;
  }
}

template <class Type>
void VArray<Type>::QuickSortPtr(int p, int r, CompareFunc compare)
{
  if (count == 0) return;
  int q;
  
  while (p < r) {
    q = PartitionPtr(p,r,compare);
    QuickSortPtr(p,q,compare);
    p = q+1;
  }
}

//-----------------------------------------------------------------------------
// @ VArray::PartitionPtr()
// ---------------------------------------------------------------------------
// Real workhorse of QuickSortPtr()....
// Requires (*Type)::operator>() for comparison.
//-----------------------------------------------------------------------------
template <class Type>
int VArray<Type>::PartitionPtr(int p, int r)
{
  Type t;
  Type x = array[p];
  int i = p-1;
  int j = r+1;
  
  while(1) {
    do {
      j--;
    } while (*array[j] > *x);
    do {
      i++;
    } while (*x > *array[i]);
    if (i < j) {
      t = array[i];
      array[i] = array[j];
      array[j] = t;
    } else {
      return j;
    }
  }
}

template <class Type>
int VArray<Type>::PartitionPtr(int p, int r, CompareFunc compare)
{
  Type t;
  Type x = array[p];
  int i = p-1;
  int j = r+1;
  
  while(1) {
    do {
      j--;
    } while (compare(array[j],x) > 0);
    do {
      i++;
    } while (compare(x,array[i]) > 0);
    if (i < j) {
      t = array[i];
      array[i] = array[j];
      array[j] = t;
    } else {
      return j;
    }
  }
}


#endif 

