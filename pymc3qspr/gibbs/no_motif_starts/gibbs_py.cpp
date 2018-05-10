include "gibbs_py.h"
#include <string>
#include <boost/foreach.hpp>
#include <algorithm>

namespace bpy = boost::python;
using namespace Gibbs;

bool Gibbs_Py::test_peptide_transfer(int key, int idx, bpy::list other_pep){
  //take a key to the map and the idx of the peptide to test, and a bpy list
  int i;
  bool match = true;
  for(i = 0; i < key; i++){
    if(bpy::extract<int>(other_pep[i]) != _peptides[key][idx][i]){
      match = false;
    }
  }
  return(match);
}

char const * Gibbs_Py::test_print(){
  //a whole bunch of test cases. this is ad-hoc and bad
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
/*  if(_peptides[5][39][4] == 11){
    return("CORRECT LAST LAST ENTRY");
  }
  else{
    return("INCORRECT LAST LAST ENTRY");
    }
  double test_val = get_tot_prob(_peptides[5][0], 5, _bg_dist, _motif_dists, _motif_class_dists_map[5][0], _motif_start_dists_map[5][0], 0, 0);
  if(test_val < 0.000001){
    return("SUCCESS");
  }
  else{
    return("FAILURE");
    }
  double test_val = _motif_start_dists_map[5][0][0];
  if(test_val - 0.5 < 0.0000001 and test_val > 0.0){
    return("CORRECTLY PASSED MOTIF_START_DISTS");
  }
  else{
    return("INCORRECTLY PASSED MOTIF_START_DISTS");
    }
  double test_val = _motif_dists[0][0][0];
  if(test_val - 0.5 < 0.0000001 and test_val > 0.0){
    return("CORRECTLY PASSED MOTIF_DISTS");
  }
  else{
    return("INCORRECTLY PASSED MOTIF_DISTS");
    }*/
  return("This is a test.");
}

void Gibbs_Py::clear_temp_dist(){
  for(i = 0; i < ALPHABET_LENGTH; i++){
    _temp_dist[i] = 0.0;
  }
  return;
}

void Gibbs_Py::time_get_tot_prob(int num_repeats, int idx){
  double local_test_prob;
  for(i = 0; i < num_repeats; i++){
    local_test_prob =   get_tot_prob(_peptides[5][idx], 5, _bg_dist, _motif_dists, _motif_class_dists_map[5][idx], _motif_start_dists_map[5][idx], -1, -1);
  }
  return;
}

bool Gibbs_Py::test_get_tot_prob(double test_prob, int idx){
//  double other_test_prob = bpy::extract<double>(test_prob);
  bool passed = false;
  double local_test_prob = get_tot_prob(_peptides[5][idx], 5, _bg_dist, _motif_dists, _motif_class_dists_map[5][idx], _motif_start_dists_map[5][idx], -1, -1);
  if(abs(local_test_prob - test_prob) < EPSILON){
    passed = true;
  }
  return(passed);
}

bool Gibbs_Py::test_rng(double test_random){
  bool passed = false;
  static boost::uniform_01<boost::mt19937> zero_one(_rng);
  double local_random = zero_one();
  if(abs(local_random - test_random) < EPSILON){
    passed = true;
  }

  return(passed);

}

void Gibbs_Py::do_bg_counts(int* peptide, int length, int start){
  for(i = 0; i < length; i++){
    if (i < start or i >= (_motif_length + start) ){
      _local_bg_counts[peptide[i]]++;
    }
  }
}

int Gibbs_Py::test_random_choice(int num_choices){
  double* weights = new double[num_choices];
  int choice;
  if(num_choices == 5){
    //special case of test
    for(i = 0; i < num_choices; i++){
      if(i!=(num_choices-1)){
	weights[i] = 0.0;
      }
      else{
	weights[i] = 1.0;
      }
    }
    choice = random_choice(num_choices, weights);
    return(choice);
  }
  else if(num_choices ==6){
    //special case of test
    for(i = 0; i < num_choices; i++){
      if(i!=(0)){
	weights[i] = 0.0;
      }
      else{
	weights[i] = 1.0;
      }
    }
    choice = random_choice(num_choices, weights);
    return(choice);
  }
  else if(num_choices == 7){
    //special case of test
    for(i = 0; i < num_choices; i++){
      if(i!=(3)){
	weights[i] = 0.0;
      }
      else{
	weights[i] = 1.0;
      }
    }
    choice = random_choice(num_choices, weights);
    return(choice);
  }
  else{
    for(i = 0; i < num_choices; i++){
      weights[i] = double(1) / double(num_choices);
    }
    choice = random_choice(num_choices, weights);
    return(choice);
  }
  
  delete [] weights;
}

int Gibbs_Py::random_choice(int num_choices, double* weights){
  //expects that the weights are normalized
  static boost::uniform_01<boost::mt19937> zero_one(_rng);
  double rando = zero_one();
  for( i = 0; i < num_choices; i++){
    if(rando < weights[i]){
      return(i);
    }
    rando -= weights[i];
  }
  return(-1);//will raise an error if we don't do it right.
}

void Gibbs_Py::update_bg_dist(){
  double bg_dist_sum = 0.0;
  //keep it normalized
  for (int i = 0; i < ALPHABET_LENGTH; i++){
    _bg_dist[i] += double(_local_bg_counts[i]);
  }
  
  for (int i = 0; i < ALPHABET_LENGTH; i++){
    bg_dist_sum += _bg_dist[i];
  }
  if(abs(bg_dist_sum - 1.0) > 0.0001){
    bg_dist_sum = 0.0;
    for (int i = 0; i < ALPHABET_LENGTH; i++){
      bg_dist_sum += _bg_dist[i];
    }
    for (int i = 0; i < ALPHABET_LENGTH; i++){
      _bg_dist[i] /= bg_dist_sum;
    }

  }
  return;
}

bpy::tuple Gibbs_Py::run(){
  /*
   * The main loop is embodied here. After calling run(), we must then pass all
   * the altered distros back to the python side of things
   */
  int i_key, key, i, j, k, step, poss_starts, motif_start, motif_class, random_idx, count;
  double motif_dists_sum;
  double uniform_pep_dist[ALPHABET_LENGTH];
  double uniform_motif_idx_dist[_motif_length];
  for(i = 0; i < ALPHABET_LENGTH; i++){
    uniform_pep_dist[i] = 1.0/double(ALPHABET_LENGTH);
  }
  for(i = 0; i < _motif_length; i++){
    uniform_motif_idx_dist[i] = 1.0/_motif_length;
  }
  int* pep;
  for(step = 0; step < _num_iters; step++){
    //loop over keys
    motif_dists_sum = 0;
    for (i_key = 0; i_key < _num_keys; i_key++){
      key = (_keys[i_key]);
      poss_starts = (key - _motif_length +1);
      for(i = 0; i < _num_motif_classes; i++){
	for(j = 0; j < _motif_length; j++){
	  for(k = 0; k < ALPHABET_LENGTH; k++){
	    _motif_counts_map[key][i][j][k] = 0;
	  }
	}
      }
      //loop over all peptides of that length
      for(i = 0; i < _lengths[key]; i++){
	count = 0;//for the L1 regularization step
	pep = _peptides[key][i];
	//randomly choose motif start
	motif_start = random_choice(poss_starts, _motif_start_dists_map[key][i]);
	do_bg_counts(pep, key, motif_start);
	update_bg_dist();
	motif_class = random_choice(_num_motif_classes, _motif_class_dists_map[key][i]);
	for (j = 0; j < _motif_length; j++){
	  int aa = pep[j+motif_start];
	  _motif_counts_map[key][motif_class][j][aa] += 1;
	  count += 1;
	}//j
	//add random noise
	for(j = 0; j < _num_random_draws; j++){
	  k = random_choice(_motif_length, uniform_motif_idx_dist);
	  random_idx = random_choice(ALPHABET_LENGTH, uniform_pep_dist);
	  _motif_counts_map[key][motif_class][k][random_idx] += 1;
	  count += 1;
	  }
	for (j = 0; j < _motif_length; j++){
	  for(k = 0; k < ALPHABET_LENGTH; k++){
	    _eta = _horizon_param / sqrt(_grad_square_sums[motif_class][j][k]);
	    _gradient = ( double(count) * ( double(count) * _motif_dists[motif_class][j][k] - double(_motif_counts_map[key][motif_class][j][k])/double(count)) ) + _alpha;//apply regularization
	    _motif_dists[motif_class][j][k] -= (_eta * _gradient > _motif_dists[motif_class][j][k] ? _motif_dists[motif_class][j][k]: _eta * _gradient );
	    _grad_square_sums[motif_class][j][k] += _gradient * _gradient;
	  }
	}
      }//i
    }//i_key
    //NORMALIZE MOTIF DISTROS
    for(i = 0; i < _num_motif_classes; i++){
      for(j = 0; j < _motif_length; j++){
	motif_dists_sum = 0.0;
	if(abs(1.0 - motif_dists_sum) > 0.0001 ){
	  motif_dists_sum = 0.0;
	  for(k = 0; k < ALPHABET_LENGTH; k++){
	    motif_dists_sum += _motif_dists[i][j][k];
	  }//k
	  for(k = 0; k < ALPHABET_LENGTH; k++){
	    _motif_dists[i][j][k] /= motif_dists_sum;
	  }//k
	}

      }//j
    }//i

    for (i_key = 0; i_key < _num_keys; i_key++){
      key = (_keys[i_key]);
      poss_starts = (key - _motif_length +1);
      for(i = 0; i < _lengths[key]; i++){
	clear_temp_dist();
	for(j = 0; j < (_num_motif_classes); j++){
	  for(k = 0; k < (poss_starts); k++){
	    _temp_dist[j] += get_tot_prob(
	      _peptides[key][i], key, _bg_dist, _motif_dists,
	      _motif_class_dists_map[key][i], _motif_start_dists_map[key][i], j, k
	      );
	  }//k
	}//j
      	for(j = 0; j < (_num_motif_classes); j++){
	  _motif_class_dists_map[key][i][j] += _temp_dist[j];
	}
      }//i
      for(i = 0; i < (_lengths[key]); i++){
	motif_dists_sum = 0;
	if(abs(1.0 - motif_dists_sum) > 0.00001){
	  motif_dists_sum = 0;
	  for(j = 0; j < _num_motif_classes; j++){
	    motif_dists_sum += _motif_class_dists_map[key][i][j];
	  }
	  for(j = 0; j < _num_motif_classes; j++){
	    _motif_class_dists_map[key][i][j] /= motif_dists_sum;
	  }

	}
      }
    }//i_key
  }//step
  //now that it's all done, return the distros!
  //first, sort by expected value (alphabetical based on weights from dists)
  std::vector<std::pair <double, int> > pairs;

  for(i = 0; i < _num_motif_classes; i++){
    motif_dists_sum = 0.0;
    for(j = 0; j < _motif_length; j++){
      for(k = 0; k < ALPHABET_LENGTH; k++){
	motif_dists_sum += (_motif_dists[i][j][k] * double(k));//end up withthe expected value
      }
    }
    pairs.push_back(std::make_pair(motif_dists_sum, i));//now we have pairs
  }
  std::sort(pairs.begin(), pairs.end());//sort alphabetically by expected value
  
  int order_idx;
  for(i = 0; i < _num_motif_classes; i++){
    order_idx = pairs[i].second;//the sorted index to use
    for(j = 0; j < _motif_length; j++){
      for(k = 0; k < ALPHABET_LENGTH; k++){//this is borked
	_other_motif_dists[i][j][k] = _motif_dists[order_idx][j][k];
      }
    }
  }
  for(i_key = 0; i_key < _num_keys; i_key++){
    key = (_keys[i_key]);
    for(i = 0; i < _lengths[key]; i++){
      for(j = 0; j < (key - _motif_length +1); j++){
	_motif_start_dists[key][i][j] = _motif_start_dists_map[key][i][j];
      }
    }
    for(i = 0; i < _lengths[key]; i++){
     
      for(j = 0; j < (_num_motif_classes); j++){
	 order_idx = pairs[j].second;//need to sort these to properly ID classes
	_motif_class_dists[key][i][j] = _motif_class_dists_map[key][order_idx][j];
      }
    }
  }
  bpy::list other_bg_dist;
  for(i = 0; i < ALPHABET_LENGTH; i++){
    other_bg_dist.append(_bg_dist[i]);
  }
  return(bpy::make_tuple(_other_motif_dists, other_bg_dist, _motif_start_dists, _motif_class_dists));
}

double Gibbs_Py::get_tot_prob(int* peptide,
			      int length,
			      double* bg_dist,
			      double*** motif_dists,//3D arr
			      double* class_dist,//specific distro we're using here
			      double* start_dist,
			      int motif_class,
			      int motif_start){
  //these are all internal variables that will be used; don't call this from python.
  /*
   * Takes in a peptide as an int array, the lenght of that peptide,
   * a double arr containing the background distro, a 3D arr of doubles containing
   * the motif distros, a double arr containing the class distro, a double arr 
   * containing the motif start distro, the current motif class (if applicable, else -1),
   * and the current motif start position (if applicable, else -1). Returns the 
   * un-normalized probability density assigned to this peptide with these params.
   */
  double prob = 0.0;
//  int i, j, k;
  if((motif_start >= 0) ){//use set value for motif_start
    if((motif_class >=0)){//use set value for motif_class
      for (i = 0; i < length; i++){
	for (j = 0; j < (length - _motif_length + 1); j++){
	  for (k = 0; k < _num_motif_classes; k++){
	    if( i < motif_start or i >= (motif_start + _motif_length)){
	      prob += bg_dist[peptide[i]] * start_dist[j] * class_dist[motif_class];
	    }
	    else{
	      prob += motif_dists[motif_class][ i - motif_start][peptide[i]] * start_dist[j] * class_dist[motif_class];
	    }
	  }//for k
	}//for j
      }//for i
    }//if(motif_class >= 0)
    else{//no motif class given, use distro
      for (i = 0; i < length; i++){
	for (j = 0; j < (length - _motif_length + 1); j++){
	  for (k = 0; k < _num_motif_classes; k++){
	    if( i < motif_start or i >= (motif_start + _motif_length)){//not in a motif
	      prob += bg_dist[peptide[i]] * start_dist[j] * class_dist[k];
	    }
	    else{//in a motif
	      prob += motif_dists[k][i - motif_start][peptide[i]] * start_dist[j] * class_dist[k];
	    }
	  }
	}
      }
    }
  }//if(motif_start >= 0)
  else{//use start_dist; no set value
    if ((motif_class >=0) ){//use set value for motif_class
      for (i = 0; i < length; i++){
	for (j = 0; j < (length - _motif_length + 1); j++){
	  for (k = 0; k < _num_motif_classes; k++){
	    if( i < j or i >= (j + _motif_length)){//not in a motif
	      prob += bg_dist[peptide[i]] * start_dist[j] * class_dist[k];
	    }
	    else{//in a motif
	      prob += motif_dists[motif_class][i - j][peptide[i]] * start_dist[j] * class_dist[motif_class];
	    }
	  }//for k
	}//for j
      }//for i
    }//if(motif_class >= 0)
    else{
      for (i = 0; i < length; i++){
	for (j = 0; j < (length - _motif_length + 1); j++){
	  for (k = 0; k < _num_motif_classes; k++){
	    if( i < j or i >= (j + _motif_length)){//not in a motif
	      prob += bg_dist[peptide[i]] * start_dist[j] * class_dist[k];
	    }
	    else{//in a motif
	      prob += motif_dists[k][i - j][peptide[i]] * start_dist[j] * class_dist[k];
	    }
	  }//for k
	}//for j
      }//for i
    }
  }
  return(prob);
}

Gibbs_Py::Gibbs_Py(bpy::dict training_peptides,
		   bpy::dict motif_counts,
		   bpy::dict motif_start_dists,
		   bpy::dict motif_class_dists,
		   bpy::list motif_dists,
		   bpy::list bg_counts,
		   bpy::list tot_bg_counts,
		   int num_iters,
		   int motif_length,
  		   int num_motif_classes,
		   int rng_seed,
		   int num_random_draws,
		   double alpha){
  _eta = 1.0;
  _num_random_draws = num_random_draws;
  _motif_counts = motif_counts;
  _peptides_dict = training_peptides;
  _motif_start_dists = motif_start_dists;
  _motif_class_dists = motif_class_dists;
  _other_motif_dists = motif_dists;
  _bg_counts = bg_counts;
  _tot_bg_counts = tot_bg_counts;
  _num_iters = num_iters;
  _motif_length = motif_length;
  _num_motif_classes = num_motif_classes;
  _local_bg_counts = new int[ALPHABET_LENGTH];
  _bg_dist = new double[ALPHABET_LENGTH];//the length of the alphabet
  _motif_dists = new double**[_num_motif_classes];
  _grad_square_sums = new double**[_num_motif_classes];
  _horizon_param = 0.005;//for now
  _gradient = 0.0;
  _alpha = alpha;
  _keys_list = training_peptides.keys();
  _keys = new int[bpy::len(_keys_list)];

  int key, length;

  _rng = boost::random::mt19937(rng_seed);

  _num_keys = bpy::len(_keys_list);

  for(i = 0; i < _num_keys; i++){
    _keys[i] = bpy::extract<int>(_keys_list[i]);
  }

  for(i = 0; i < ALPHABET_LENGTH; i++){
    _local_bg_counts[i] = 0;
    _bg_dist[i] = 1.0/double(ALPHABET_LENGTH);
  }

  for (i = 0; i < _num_keys; i++){
    key = (_keys[i]);
    length = bpy::len(training_peptides[key]);
    _lengths[key] = length;
    _counts[key] = length;
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
    _grad_square_sums[i] = new double*[_motif_length];
    for( j = 0; j < _motif_length; j++){
      _motif_dists[i][j] = new double[ALPHABET_LENGTH];
      _grad_square_sums[i][j] = new double[ALPHABET_LENGTH];
      for(k = 0; k < ALPHABET_LENGTH; k++){
	_motif_dists[i][j][k] = 1.0/double(ALPHABET_LENGTH);
	_grad_square_sums[i][j][k] = 1.0;
      }
    }
  }

  for( i = 0; i < _num_keys; i++){
    key = ((_keys[i]));
    length = _lengths[key];
    _motif_start_dists_map[key] = new double*[length];
    _motif_class_dists_map[key] = new double*[length];
    _motif_counts_map[key] = new int**[_num_motif_classes];
    for( j = 0; j < length; j++){
      _motif_start_dists_map[key][j] = new double[key - _motif_length +1];
      _motif_class_dists_map[key][j] = new double[_num_motif_classes];

      for( k = 0; k < key - _motif_length +1; k++){
	_motif_start_dists_map[key][j][k] = bpy::extract<double>(motif_start_dists[key][j][k]);
      }
      for(k = 0; k < _num_motif_classes; k++){
	_motif_class_dists_map[key][j][k] = bpy::extract<double>(motif_class_dists[key][j][k]);
      }
    }//j
    for(j = 0; j < num_motif_classes; j++){
      _motif_counts_map[key][j] = new int*[_motif_length];
      for(k = 0; k < _motif_length; k++){
	_motif_counts_map[key][j][k] = new int[ALPHABET_LENGTH];
	for(h = 0; h < ALPHABET_LENGTH; h++){
	  _motif_counts_map[key][j][k][h] = 0;
	}
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

  int key, length, i, j;

  for( i = 0; i < _num_keys; i++){
    key = (_keys[i]);
    length = _lengths[key];
    for( j = 0; j < length; j++){
	delete [] _motif_start_dists_map[key][j];
	delete [] _motif_class_dists_map[key][j];
    }
    for(j = 0; j < _num_motif_classes; j++){
      delete [] _motif_counts_map[key][j];
    }
  }
}
