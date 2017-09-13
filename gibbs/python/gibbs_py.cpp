#include "gibbs_py.h"
#include <string>
#include <vector>
#include <boost/foreach.hpp>

namespace bpy = boost::python;
using namespace Gibbs;

char const * Gibbs_Py::test_print(){
  /*if(_num_iters == 3000){
    return("NRUNS INITIALIZATION WORKS");
  }
  else{
    return("NRUNS INITIALIZATION FAILED");
    }*/
/*  if(_peptides.has_key(5)){
    if(bpy::len(_peptides[5][0]) == 5){
      return("CORRECT LENGTH");
    }
    else{
      return("INCORRECT LENGTH");
    }
  }
  else{
    return("KEY NOT RECOGNIZED");
    }*/
/*  if(_peptides[5][0][0] == 0){
    return("CORRECT FIRST FIRST ENTRY");
  }
  else{
    return("INCORRECT FIRST FIRST ENTRY");
    }*/
  return("This is a test.");
}

double Gibbs_Py::Gibbs_Py::get_tot_prob(int* peptide,
					double* bg_dist,
					double*** motif_dists,//3D arr
					double* class_dist,//specific distro we're using here
					double* start_dist,
					int motif_class,
					int motif_start){
  //these are all internal variables that will be used; don't call this from python.
  double prob = 0.0;
  
  return(prob);
}

Gibbs_Py::Gibbs_Py(bpy::dict training_peptides,
		   bpy::dict motif_counts,
		   bpy::dict motif_start_dists,
		   bpy::dict motif_class_dists,
		   bpy::list bg_counts,
		   bpy::list tot_bg_counts,
		   int num_iters,
		   int motif_length,
  		   int num_motif_classes){
  _motif_counts = motif_counts;
  _motif_start_dists = motif_start_dists;
  _motif_class_dists = motif_class_dists;
  _bg_counts = bg_counts;
  _tot_bg_counts = tot_bg_counts;
  _num_iters = num_iters;
  _motif_length = motif_length;
  _num_motif_classes = num_motif_classes;

  _bg_dist = new double[ALPHABET_LENGTH];//the length of the alphabet
  _motif_dists = new double**[_num_motif_classes];

  bpy::list keys = training_peptides.keys();
  int key, length, i, j, k;

  for (i = 0; i < bpy::len(training_peptides.keys()); i++){
    key = bpy::extract<int>(keys[i]);
    length = bpy::len(training_peptides[key]);
    _peptides[key] = new int*[length];
    for(j = 0; j < length; j++){
      _peptides[key][j] = new int[key];//keyed by length
      for(k = 0; k < key; k++){
	_peptides[key][j][k] = bpy::extract<int>(training_peptides[key][j][k]);
      }
    }
  }

  for (i = 0; i < _num_motif_classes; i++){
    _motif_dists[i] = new double*[_motif_length];
  }
  for ( i = 0; i < _num_motif_classes; i++){
    for( j = 0; j < _motif_length; j++){
      _motif_dists[i][j] = new double[ALPHABET_LENGTH];
    }
  }

  for( i = 0; i < bpy::len(training_peptides.keys()); i++){
    key = (bpy::extract<int>(keys[i]));
    length = bpy::len(training_peptides[key]);
    _motif_start_dists_map[key] = new double*[length];
    _motif_class_dists_map[key] = new double*[length];
    for( j = 0; j < length; j++){
      _motif_start_dists_map[key][j] = new double[key - _motif_length +1];
      _motif_class_dists_map[key][j] = new double[_num_motif_classes];
      for( k = 0; k < key - _motif_length +1; k++){
	_motif_start_dists_map[key][j][k] = 0.0;
      }
      for(k = 0; k < _num_motif_classes; k++){
	_motif_class_dists_map[key][j][k] = 0.0;
      }
    }
  }
  //AT THE END OF THE LOOP, REPLACE DICT CONTENTS WITH THE UPDATED DISTROS

  
};

Gibbs_Py::~Gibbs_Py(){
  if(_bg_dist){
    delete [] _bg_dist;
  }
  if(_motif_dists){
    delete [] _motif_dists;    
  }

  std::pair<int, int**> item;
  std::vector<int> delete_keys;
  BOOST_FOREACH(item, _peptides){
    delete_keys.push_back(item.first);
  }
    
  int key, length, i, j;

  for( i = 0; i < delete_keys.size(); i++){
    key = delete_keys[i];
    length = sizeof(_peptides[key])/sizeof(int*);
    for( j = 0; j < length; j++){
	delete [] _motif_start_dists_map[key][j];
	delete [] _motif_class_dists_map[key][j];
    }
  }
}
