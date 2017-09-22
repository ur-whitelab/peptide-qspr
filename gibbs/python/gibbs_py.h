#ifndef GIBBS_PY_H_
#define GIBBS_PY_H_

#define ALPHABET_LENGTH 20
#define EPSILON 0.0001

#include <boost/python.hpp>
#include <boost/random.hpp>
#include <map>
#include <vector>

namespace bpy = boost::python;

namespace Gibbs{
  class Gibbs_Py{
  public:
    //keep public copies of these to pass back to python at the end so we can plot
    bpy::dict _motif_counts;
    bpy::dict _peptides_dict;
    bpy::dict _motif_start_dists;
    bpy::dict _motif_class_dists;
    bpy::list _other_motif_dists;
    bpy::list _bg_counts;
    bpy::list _tot_bg_counts;
    bpy::list _keys;
    int _num_iters;
    int _motif_length;
    int _num_motif_classes;

    Gibbs_Py(bpy::dict training_peptides,
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
	     int num_random_draws);
//    Gibbs_Py(const Gibbs_Py& that);//disable copy constructor

    ~Gibbs_Py();//destroys dynamically allocated stuff

    char const* test_print();//for testing
    bool test_peptide_transfer(int key, int idx, bpy::list other_pep);
    bool test_get_tot_prob(double test_prob, int idx);
    bool test_rng(double test_random);
    int test_random_choice(int num_choices);
    void time_get_tot_prob(int num_repeats, int idx);

    bpy::tuple run();//runs through the number of steps passed in at creation

  private:
    int _num_random_draws;
    double* _bg_dist;//background distro
    int* _local_bg_counts;//the internal background counts
    double***  _motif_dists;//the 3D array of motif distros
    std::map<int, double**> _motif_start_dists_map;
    std::map<int, double**> _motif_class_dists_map;//the map of motif class distros
    std::map<int, int***> _motif_counts_map;//map to raw counts of AA occurrences
    std::map<int, int**> _peptides;
    std::map<int, int> _counts;//keyed same as _peptides, counts number of each length
    int _num_keys; //the number of different lenghts of peptide we have
    double _temp_dist[ALPHABET_LENGTH];//for holding updates before applying,
    //so get_tot_prob() works properly

    void do_bg_counts(int* peptide, int length, int start);
    void update_bg_dist();
    int random_choice(int num_choices, double* weights);
    void clear_temp_dist();

    boost::random::mt19937 _rng;

    //private because we can't call from python due to C++ data types
    double get_tot_prob(int* peptide,
			int length,
			double* bg_dist,
			double*** motif_dists,
			double* class_dist,//specific distro we're using here
			double* start_dist,
			int motif_class,
			int motif_start);

  };
}

#endif//GIBS_PY_H_
