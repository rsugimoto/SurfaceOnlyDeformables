
// 
// Copyright Emin Martinian, emin@allegro.mit.edu, 1998
// Permission granted to copy and distribute provided this
// comment is retained at the top.
//
// $Id: my_hash_map.H,v 1.13 2002/11/11 16:40:18 emin Exp $
//
//
// *** IMPORTANT: When you want to compile with debugging assertions
//                turned off, #define NDEBUG .If the NDEBUG preprocessor
//		  token is not defined then debugging assertion
//		  checks are turned on by default.
//

#ifndef EMIN_HASH_TABLE
#define EMIN_HASH_TABLE

// --------------------------------------------------------------
//
// Introduction:
//
// This file is desigend to implement a generic hash table class
// called "my_hash_map".  The my_hash_map class is based upon the
// hash_map class in the UNIX implementation of the
// Standard Template Library.  For some incomprehensible reason
// the hash_map class is not part of the STL standard so some 
// compilers (such as the Microsoft compiler) don't implement
// hash_map.  Basically, the my_hash_map class was designed to 
// have the same interface as the hash_map class from STL, but
// there are currently a few slight differences which I will
// eventually fix.
//
// How To Use: Setting things up
//
// Using the my_hash_map class is pretty simple.  First decide 
// the key type and the value type.  For example, assume you want
// a hash table indexed with integer keys with characters as the
// value.  The first thing you need to do is create a hashing
// object.  The hashing object will implement the hash function
// for your keys.  The hashing object has to be an object which 
// implements hash functions.  For most efficient preformance
// your functor should implement operator() and SecondHashValue,
// but if you want you can just implement operator().
//
// The operator() member function should take the key as input and
// return an integer as output.  Ideally the hash function will
// have the property that different keys are mapped to different
// hash results, but this is not absolutely essential.  The
// SecondHashValue member function is another hash function which
// takes the key as input, but it MUST return an ODD integer as
// the output.  The reason SecondHashValue must return and ODD
// integer is described in the Design Overview sectoion.
//
// The GENERIC_HASH function is a generic hash function which
// hashes on integers.  You can use this function if your
// keys are integers, or you can use it to build more
// complicated hash functions.
//
// How To Use: constructor
//
// Assume we have defined a hashing object called IntHasher, then
// we can create a hash table as follows:
//
//		IntHasher hashObject;
//		my_hash_map<int, char, IntHasher> myTable(10,hashObject);
//
// The first argument in the template specification is "int" 
// which means that the keys will be integers.  The second
// argument in the template specifcation is "char" meaning that
// the values will be characters.  Finally, the third argument
// is "IntHasher" meaning an object of the hashObject class will
// provide the necessary hashing.  
//
// The first argument of the constructor is the initial size of the
// hash table.  The second argument to the constructor is an instance
// of the IntHasher class which will provide the necessary hashing
// services.  The initial size will be rounded up to the nearest power
// of 2.  Therefore we could have equivalently put 11, 13, or 16 and
// the size would still be 16.  You don't really have to worry about
// the size because the my_hash_map classes will automatically grow
// itself when necessary.  Thus even if you start with size 16, and
// then put in 50 items everything will be fine.
//
// You can control the resize behavior by providing a third argument
// corresponding to the resize ratio.  Whenver the ratio of items in
// the table to the table size exceeds the resize ratio, the hash
// table expands.  The default resize ratio is .5 which means that the
// table expands once it becomes over half full.  You probably
// shouldn't mess with the resize ratio unless you understand how this
// affects hash table and search times.  See the
// _Introduction_To_Algorithms_ book by Cormen, Leisserson, and Rivest
// for such a discussion.
//
// How To Use: inserting
//
// To insert a key and value you can either use the insert
// member function or the [] operator.  For example, 
// the following statements would insert a value of 'c' or 'd' 
// into the table for a key of 3:
//
//		myTable.insert(3,'c');
//		myTable[3] = 'd';
//
// Note that the hash table will only store 1 value for each key.
// Thus the second statement above will replace the value of 'c'
// with 'd' for the key 3.
//
// How To Use: searching
//
// The find(...) function can be used to look for a key and value
// pair in the hash table.  Calling myTable.find(someKey) returns
// a HashTableIterator.  Iterator are basically like generalized
// pointers.  The iterator idea is from the Standard Template
// Library.  If someKey exists in myTable, then the iterator 
// returned will point to the corresponding key-value pair.  If
// someKey is not in myTable the the iterator returned will match
// the iterator returned by myTable.end().  The following code
// fragment gives an example of how search for an item using
// iterators:
//
// HashTableIterator<int,char,IntHasher> i = myTable.find(3);
// if (i < myTable.end()) {
//		printf("found item with key = %i ",i->first);
//		printf("and value %c\n",i->second);
// } else {
//		printf("no match in table\n");
// }
//
// You can also say my_hash_map<int,char,IntHasher>::iterator
// instead of HashTableIterator<int,char,IntHasher>.  The first
// syntax corresponds to the STL convention.  Alternativly
// you can use a typedef statement.
//
// How To Use: Deleting
//
// To delete an item from the table you can use the Delete
// member function.  Calling myTable.Delete(someKey) will attempt 
// to delete the pair with key someKey.  If someKey is found in
// the table then it is deleted and true is returned, otherwise
// false is returned and the table is not modified.
//
// How To Use: for_each
//
// One of the main uses of iterators is that you can do an
// operation for each item in the hash table.  If you are
// using the Standard Template Library you can simply use
// the for_each function.  Otherwise you can use a code
// fragment like the one below:
//
// for(HashTableIterator<int,char,IntHasher> i = myTable.begin();
//      i < myTable.end(); i++) {
//		printf("key: %i, value %c\n",i->first,i->second);
// }
//
// Design Overview:
//
//
// The my_hash_map is implemented as a quadratic probing hash
// table.  In a quadratic probing table we use two hash functions
// for each key.  The first hash function tells us a starting
// point in the table to insert the new record.  If there is
// already another record in that spot (e.g. because another
// key hashed to the same spot), then we use the second hash
// function to check other spots in the table.  To make sure
// that the entire hash table is searched when necessary the
// value returned by the second hash function must be 
// relatively prime to the size of the hash table.  This is
// accomplished by making the hash table size a power of 2 and
// requiring the second hash function return an odd value.
//
// 
//
// For an example of how to use the my_hash_map class see
// the hash_test.cpp file.

//
// Efficiency Comments: If you do a lot of insertion and deletion
// without resizing the table, then it is possible for the hash table
// to get into an inefficient state.  This is because when stuff is
// deleted from the hash table we mark the location as deleted instead
// of empty due to the requirements of the quadratic probing
// algorithm.  Consequently if lots of cells get marked as deleted
// instead of empty then inserting and searching will become slow (as
// slow as O(n)).  Therefore if you are using the hash table in an
// application where you do lots of inserting and deleting it might be
// good to periodically resize or otherwise clear the table.  When I
// get some time I will put in stuff to automatically keep track of
// the fraction of deleted cells and automatically clean up the table
// when there are too many deleted cells.

//
// 
// TODO:
//
// *   Make the hash table keep track of how many deleted cells exist
//     and automatically clean up if there are too many.  See the
//     efficiency comment above.
//
// *	Right now the const_iterator is the same as an iterator
//	we should fix this so that const_iterator implements const-ness.
//


#include <stdio.h>
#include <stdlib.h>
//#include <iostream.h>
//#include "my_pair.h"
#include "_pair.h"

/* The following two constants are used to build a hash function based
 * on the multiplication method described in the book
 * _Introduction_To_Algorithms_ by Cormen, Leisserson, and Rivest. */

static const double g_HASH_CONSTANT =  0.6180339887; /* (sqrt(5) -1)/2 */
static const double g_HASH_SEED  = 1234567;  /* could also make this
                                                randomly generated */

inline int GENERIC_HASH(const int dataToHash) {
  return ( (int) ( g_HASH_SEED * ( g_HASH_CONSTANT * ( dataToHash) - 
				   (int) (g_HASH_CONSTANT*( dataToHash) ))));
}


#define EMPTY_CELL 0
#define VALID_CELL 1
#define DELETED_CELL 2

// Define Hash_Assert macro to check assumptions during debugging.
#ifdef NDEBUG
#define Hash_Assert(a , b) ;
#else
#define Hash_Assert(a , b) if (!a) \
{ fprintf(stderr,"Error: Hash_Assertion '%s' failed.\n%s\n",#a,b); abort(); }
#endif

// Define ExitProgramMacro to abort if something wierd happens
#define ExitProgramMacro( a ) {fprintf(stderr, a ); abort();}

#define MHM_TEMPLATE <class HashType, class HashValue, class Hasher, class EqualityComparer>  
#define MHM_TEMP_SPEC <HashType,HashValue,Hasher,EqualityComparer> 
#define MHM_TYPE_SPEC my_hash_map MHM_TEMP_SPEC

template<class HashType> class MyHasherModel {
public:
  virtual int operator()(const HashType &) const = 0;
  inline int SecondHashValue(const HashType & key) const {
    return (operator()(key) << 1) + 1;
  }
};

template MHM_TEMPLATE class my_hash_map;

template MHM_TEMPLATE class HashTableIterator {
  friend class MHM_TYPE_SPEC;
public:
  HashTableIterator(MHM_TYPE_SPEC const & hashTable)
    :_table(&hashTable), _currentIndex(0)
  {
  }

  HashTableIterator(MHM_TYPE_SPEC const & hashTable, const int startingIndex)
    :_table(&hashTable), _currentIndex(startingIndex)
  {
  }

  HashTableIterator()
    :_table(NULL), _currentIndex(0)
  {
  }

  inline my_pair<const HashType, HashValue>* GetCurrentItem() 
  {
    while(_currentIndex < _table->maxLength)
      if (VALID_CELL == _table->tableStatus[_currentIndex])
        return _table->tableData[_currentIndex];
      else
	_currentIndex++;
    return NULL;
  }

  inline my_pair<const HashType, HashValue>* operator++(int) 
    // postfix operator
  {
    my_pair<const HashType, HashValue> * result = NULL;
    for(; _currentIndex < _table->maxLength; _currentIndex++)
      if (VALID_CELL == _table->tableStatus[_currentIndex]) {
	result = _table->tableData[_currentIndex++];
	break;
      }
    for(; _currentIndex < _table->maxLength; _currentIndex++)
      if (VALID_CELL == _table->tableStatus[_currentIndex]) 
	break;
    return result;
  }

  inline my_pair<const HashType, HashValue> & operator*() {
    return *(GetCurrentItem());
  }

  inline my_pair<const HashType, HashValue> * operator->() {
    return & (operator*());
  }

  inline my_pair<const HashType, HashValue> * operator++() 
    // prefix operator
  {
    ++_currentIndex;
    while(_currentIndex < _table->maxLength)
      if (VALID_CELL == _table->tableStatus[_currentIndex]) 
        return _table->tableData[_currentIndex];
      else 
	++_currentIndex;
    return NULL;
  }

  inline bool operator<(HashTableIterator MHM_TEMP_SPEC
			const& other) const {
    return _currentIndex < other._currentIndex;
  }

  inline bool operator!=(HashTableIterator MHM_TEMP_SPEC const& other) const {
    return _currentIndex != other._currentIndex;
  }

  inline bool operator>(HashTableIterator MHM_TEMP_SPEC const& other) const{
    return _currentIndex > other._currentIndex;
  }

  inline bool operator==(HashTableIterator MHM_TEMP_SPEC const& other) const{
    return _currentIndex == other._currentIndex;
  }

private:
  const MHM_TYPE_SPEC * _table;
  unsigned int _currentIndex;
};

template MHM_TEMPLATE
class my_hash_map {
public:
  typedef my_pair<const HashType, HashValue> pairType;
  typedef HashTableIterator MHM_TEMP_SPEC iterator;
  typedef HashTableIterator MHM_TEMP_SPEC const_iterator;

  friend class HashTableIterator MHM_TEMP_SPEC;

  my_hash_map(unsigned int tableSize, Hasher const & hashFunctor, 
	      const EqualityComparer &, float newResizeRatio = .5);  

  ~my_hash_map();

  HashValue & operator[](HashType const &);

  pairType * insert(HashType const &,HashValue const &);

  bool Delete(HashType const &);

  iterator Search(HashType const &) const;
  iterator find(HashType const & key) const {return Search(key);}

  iterator InsertWithoutDuplication(HashType const &, HashValue const &,int*);

  void Resize(unsigned int);
  //ostream & operator<<(ostream &) const;

  void clear();

  inline const iterator begin() const { 
    // Note that we can't just return an iterator pointing at tableData[0]
    // because tableData[0] might not be a VALID_CELL.  So if the table
    // is empty we want begin() == end().
    if (currentLength == 0)
      return _end;
    else
      return _begin; 
  }

  inline const iterator end() const { 
    return _end; 
  }

  inline int empty() const {return Empty();}
  inline int Empty() const { return currentLength == 0; }
  inline int NotEmpty() const { return currentLength != 0; }

  inline unsigned int MaxSize() const { return maxLength; }

  inline unsigned int size() const { return currentLength; }
  inline unsigned int Size() const { return currentLength; }

  inline float GetResizeRatio() const { return resizeRatio; }
  inline void SetResizeRatio(float a) { 
    resizeRatio=a; maxLengthTIMESresizeRatio = (int)(maxLength*a);}
private:
  void AssignToCell(const int ,pairType* );
  inline void IncrementNumDeletedCells() {
    if (++numDeletedCells >= maxLengthTIMESresizeRatio) Resize(maxLength);
  }

  inline void DecrementNumDeletedCells() {
    numDeletedCells--;
    //Hash_Assert( (numDeletedCells >= 0), "numDeletedCells went below 0" );
  }

  float resizeRatio;
  unsigned int maxLength;
  unsigned int currentLength;
  unsigned int numDeletedCells;
  unsigned int maxLengthTIMESresizeRatio;
  pairType** tableData;
  char * tableStatus;
  iterator _begin;
  iterator _end;
  Hasher	_hashFunctor;
  EqualityComparer _equalityFunctor;
};


// 
// This stuff would normally go in a .cc file but since it is a
// template and everything needs to see the definitions we include
// it in the .H file
//
// ***************************************************************

static inline unsigned int RoundUpToPowerOfTwo(unsigned int x) {
  unsigned int returnValue = 1;
  for (; x > returnValue; returnValue*=2) ; // ';' is body of for loop
  return returnValue;
}

/****** start my_hash_map functions ********/

// tableSize must be a power of 2 and SecondHashValue() must return an
// odd positve number.  This is to insure that the tableSize and the
// SecondHashValue() are relatively prime.  Otherwise the entire hash
// table might not be searched in Insert, Delete or Search (unless you
// know that SecondHashValue() will always return something relatively
// prime to tableSize.
template MHM_TEMPLATE my_hash_map MHM_TEMP_SPEC::my_hash_map
(unsigned int tableSize, const Hasher & hashFunctor, 
 const EqualityComparer & equalityFunctor, float newResizeRatio)
  : resizeRatio(newResizeRatio) ,
    maxLength(RoundUpToPowerOfTwo(tableSize)) , 
    currentLength(0) , numDeletedCells(0) ,
    maxLengthTIMESresizeRatio((int)(maxLength*resizeRatio)) ,
    tableData( new pairType*[maxLength]),
    tableStatus( new char[maxLength]), 
    _begin( HashTableIterator MHM_TEMP_SPEC (*this,0)),
    _end(  HashTableIterator MHM_TEMP_SPEC (*this,maxLength)),
    _hashFunctor(hashFunctor),
    _equalityFunctor(equalityFunctor)
{
  for (unsigned int i = 0; i < maxLength; i++) 
    tableStatus[i] = EMPTY_CELL;
}

template MHM_TEMPLATE
void my_hash_map MHM_TEMP_SPEC::AssignToCell
(const int hashLocation, pairType* newPair) {
  tableStatus[hashLocation] = VALID_CELL;
  tableData[hashLocation] = newPair;
  currentLength++;
}

template MHM_TEMPLATE
HashValue & my_hash_map MHM_TEMP_SPEC::operator[]
(HashType const & key) {
  int didInsert;
  my_hash_map MHM_TEMP_SPEC::iterator i =
    InsertWithoutDuplication(key,HashValue(),&didInsert);
  return i->second;
}

//  If currentLength * resizeRatio >= maxLength do Resize(maxLength *
//  resizeRatio).  Otherwise key,value is inserted in the hash table
//  even if it already exists in the table.  Also a pointer to the
//  pairType that is inserted into the hash table is returned.  This
//  feature is explicitly used in the implementation of
//  operator[]. For inserting only unique elements use
//  InsertWithoutDuplication.  

template MHM_TEMPLATE  typename MHM_TYPE_SPEC::pairType *
my_hash_map MHM_TEMP_SPEC::insert
(HashType const & key, HashValue const & value) 
{
  int didInsert;
  my_hash_map MHM_TEMP_SPEC::iterator i =
    InsertWithoutDuplication(key,value,&didInsert);
  if (didInsert) {
    return i.GetCurrentItem();
  } else {
    delete tableData[i._currentIndex];
    pairType*const newPair = new pairType(key,value);
    tableData[i._currentIndex] = newPair;
    return newPair;
  }
}

// If an item matching key is already in the table, then an iterator
// point to that item is returned and the table is not modified and
// *didInsert is set to 0.  Otherwise an iterator pointing to the
// newly inserted item is returned and *didInsert is set to 1.

template MHM_TEMPLATE typename my_hash_map MHM_TEMP_SPEC::iterator 
my_hash_map MHM_TEMP_SPEC::InsertWithoutDuplication
(HashType const & key, HashValue const & value, int* didInsert) 
{
  int firstDeletedLocation = -1;
  *didInsert = 0;
  if (currentLength >= maxLengthTIMESresizeRatio)
    Resize((int)(maxLength/resizeRatio));
  int hashIncrement;
  int hashLocation = _hashFunctor.operator()(key)%maxLength;
  unsigned int timesInLoop = 0; 
  Hash_Assert( (hashLocation >= 0) ,
	  "An instance of HashType returned a negative value");
  switch(tableStatus[hashLocation]) {
  case EMPTY_CELL:
    {
      AssignToCell(hashLocation,new pairType(key,value));
      *didInsert = 1;
      return iterator(*this,hashLocation);
    }
  break;
  case DELETED_CELL:
    {
      firstDeletedLocation = hashLocation;
    }
  break;
  case VALID_CELL:
    {
      if (_equalityFunctor(tableData[hashLocation]->first,key))
	return iterator(*this,hashLocation);
    }
  break;
  default:
    {
      ExitProgramMacro("Wrong Status in my_hash_map::insert(...)");
    }
  break;
  }

  hashIncrement = _hashFunctor.SecondHashValue(key) ;
  Hash_Assert( ( hashIncrement % 2) != 0, 
	  "Even value returned by SecondHashValue()");

  for(;;) {
    hashLocation = (hashLocation + hashIncrement)%maxLength;
   
    switch(tableStatus[hashLocation]) {
    case EMPTY_CELL:
      {
	if (firstDeletedLocation != -1)  
	  hashLocation = firstDeletedLocation;
	AssignToCell(hashLocation,new pairType(key,value));
	*didInsert = 1;
	return iterator(*this,hashLocation);
      }
    break;
    case DELETED_CELL:
      {
	if (firstDeletedLocation == -1) 
	  firstDeletedLocation = hashLocation;
      }
    break;
    case VALID_CELL:
      {
	if (_equalityFunctor(tableData[hashLocation]->first, key))
	  return iterator(*this,hashLocation);
      }
    break;
    default:
      {
	    ExitProgramMacro("Wrong Status in my_hash_map::Insert");
      }
    break;
    }
    if (++timesInLoop > maxLength) {
      // We searched the entire table and didn't find a matching key or
      // an empty cell.  This means that we are indeed inserting without
      // duplication.  It just happened that lots of stuff was already
      // in the hash table but got deleted.
      Hash_Assert( (firstDeletedLocation != -1),
	"insert: searched entire table without good reason");
      DecrementNumDeletedCells();
      AssignToCell(hashLocation,new pairType(key,value));
      *didInsert = 1;
      return iterator(*this,hashLocation);
    }
  }
}

//  Searches for searchInfo in the table and returns an iterator
//  pointing to the match if possible and end() otherwise.
template MHM_TEMPLATE typename
my_hash_map MHM_TEMP_SPEC::iterator 
my_hash_map MHM_TEMP_SPEC::Search
(HashType const & key) const
{
  int hashIncrement;
  int hashLocation = _hashFunctor.operator()(key)%maxLength;
#ifndef NDEBUG
  unsigned int timesInLoop = 0; // used for checking assertions
#endif
  switch(tableStatus[hashLocation]) {
  case EMPTY_CELL:
    {
      return end();
    }
  break;
  case VALID_CELL:
    {
      if (_equalityFunctor(tableData[hashLocation]->first,key))
	return iterator(*this,hashLocation);
    }
  break;
  }
  hashIncrement = _hashFunctor.SecondHashValue(key);
  Hash_Assert( (hashIncrement % 2) != 0 ,
	  "even value returned by SecondHashValue()");
  for(;;) {
    hashLocation = ( hashLocation + hashIncrement ) % maxLength;
      switch(tableStatus[hashLocation]) {
      case EMPTY_CELL:
	{
	  return end();
	}
      break;
      case VALID_CELL:
	{
	  if (_equalityFunctor(tableData[hashLocation]->first,key))
	    return iterator(*this,hashLocation);
	}
      break;
      }
      Hash_Assert((++timesInLoop < maxLength) ,
		  "searched entire hash table and still going in Search(...)");
  }
}

//  Removes deleteInfo from the hash table if it exists and does
//  nothing if the item is not in the hash table.  Returns true if the
//  item to be deleted was found in the table.

template MHM_TEMPLATE
bool my_hash_map MHM_TEMP_SPEC::Delete(HashType const & key) 
{
  int hashIncrement;
  int hashLocation = _hashFunctor.operator()(key)%maxLength;
#ifndef NDEBUG
  unsigned int timesInLoop = 0; // used for checking assertions
#endif
  switch(tableStatus[hashLocation]) {
  case EMPTY_CELL:
    {
      return false;
    }
  break;
  case VALID_CELL:
    {
      if (_equalityFunctor(tableData[hashLocation]->first,key)) {
	tableStatus[hashLocation] = DELETED_CELL;
	delete tableData[hashLocation];
	currentLength--;
	IncrementNumDeletedCells();
	return true;
      }
    }
    break;
  }
  hashIncrement = _hashFunctor.SecondHashValue(key);
  Hash_Assert( (hashIncrement > 0),
	  "negative value returned by SecondHashValue()");
  Hash_Assert( (hashIncrement % 2) != 0 ,
	  "even value returned by SecondHashValue()");
  for(;;) {
    hashLocation = ( hashLocation + hashIncrement ) % maxLength;
    switch(tableStatus[hashLocation]) {
    case EMPTY_CELL :
      {
	return false;
      }
    break;
    case VALID_CELL :
      {
	if (_equalityFunctor(tableData[hashLocation]->first, key)) {
	  tableStatus[hashLocation] = DELETED_CELL;
	  delete tableData[hashLocation];
	  currentLength--;
	  IncrementNumDeletedCells();
	  return true;
	}
      }
    break;
    case DELETED_CELL:
      {
      }
    break;
    default :
      {
	ExitProgramMacro("Wrong type in my_hash_map::Delete");
      }
    break;
    }
    Hash_Assert((++timesInLoop < maxLength) ,
		"searched entire hash table and still going in Delete(...)");
  }
}

/*
template MHM_TEMPLATE
ostream & my_hash_map MHM_TEMP_SPEC::operator<<
(ostream &s) const
{
  int k = 0;
  while ( k < maxLength) {
    s << "Location " << k << ": ";
    switch(tableStatus[k]) {
    case EMPTY_CELL: 
      {
	s << "EMPTY_CELL" << endl;
      } 
    break;
    case DELETED_CELL : 
      {
	s << "DELETED_CELL" << endl;
      }
    break;
    case VALID_CELL :
      {
	s << "VALID_CELL : " << endl;
	s << (*tableData[k]);
      }
    break;
    default :
      {
	s << "unknown type of cell : " << tableStatus[k] << endl;
      }
    break;
    }
    k++;
  }
  return s;
}
*/

template MHM_TEMPLATE
my_hash_map MHM_TEMP_SPEC::~my_hash_map() 
{
  for (unsigned int i = 0; i < maxLength; i++)
    if (tableStatus[i] == VALID_CELL)
      delete tableData[i];
  
  delete [] tableData;
  delete [] tableStatus;
}

template MHM_TEMPLATE
void my_hash_map MHM_TEMP_SPEC::Resize(unsigned int newMaxSize) 
{
  newMaxSize = RoundUpToPowerOfTwo(newMaxSize);
  Hash_Assert((newMaxSize >= currentLength),
	 "Resize called with newMaxSize < currentLength !");
#ifdef WARN_WHEN_RESIZING
  cerr << "Warning: Resize(" << newMaxSize <<") called when "<<endl;
  cerr << "resizeRatio = " << resizeRatio << endl;
  cerr << "currentLength = " << currentLength <<endl;
  cerr << "maxLength = " <<  maxLength << endl;
  cerr << "maxLengthTIMESresizeRatio = " << maxLengthTIMESresizeRatio << endl;
  if (newMaxSize == maxLength) {
    cerr << "This resize is really doing defragmentation not resizing." <<endl;
  }
#endif
  pairType** oldTableData = tableData;
  char * oldTableStatus = tableStatus;
  int oldMaxSize = maxLength;
  maxLength = newMaxSize;
  _end._currentIndex = maxLength;
  tableData = new pairType*[maxLength];
  tableStatus = new char[maxLength];
  numDeletedCells = 0;
  for (unsigned int i = 0; i < maxLength; i++) 
    tableStatus[i] = EMPTY_CELL;

  currentLength = 0;
  for (int k = 0; k < oldMaxSize ; k++) 
    if (VALID_CELL == oldTableStatus[k]) {
      insert(oldTableData[k]->first,oldTableData[k]->second);
      delete oldTableData[k];
    }
  delete [] oldTableData;
  delete [] oldTableStatus;
  maxLengthTIMESresizeRatio = ((unsigned int) (maxLength*resizeRatio));
}

template MHM_TEMPLATE
void my_hash_map MHM_TEMP_SPEC::clear()
{
  for (unsigned int i = 0; i < maxLength; i++) {
    if (tableStatus[i] == VALID_CELL)
      delete tableData[i];
    tableStatus[i] = EMPTY_CELL;
  }
  currentLength = 0;
  numDeletedCells = 0;
}

#endif



