// 
// Copyright Emin Martinian, emin@allegro.mit.edu, 1998
// Permission granted to copy and distribute provided this
// comment is retained at the top.
//

template <class TF, class TS> 
struct my_pair {
public:
  my_pair(TF const & the_first, TS const & the_second)
    :first(the_first), second(the_second)
    {}
  
  TF first;
  TS second;
};
