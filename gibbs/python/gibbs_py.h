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
	     bpy::list bg_counts,
	     bpy::list tot_bg_counts,
	     int num_iters,
	     int motif_length,
	     int num_motif_classes,
	     int rng_seed);
//    Gibbs_Py(const Gibbs_Py& that);//disable copy constructor

    ~Gibbs_Py();//destroys dynamically allocated stuff

    char const* test_print();//for testing
    char const* test_get_tot_prob(double test_prob, int idx);
    char const* test_rng(double test_random);
    char const* test_random_choice(int num_choices);
    void time_get_tot_prob(int num_repeats, int idx);

    void run();//runs through the number of steps passed in at creation

  private:
    double* _bg_dist;//background distro
    int* _local_bg_counts;//the internal background counts
    double***  _motif_dists;//the 3D array of motif distros
    std::map<int, double**> _motif_start_dists_map;
    std::map<int, double**> _motif_class_dists_map;//the map of motif class distros
    std::map<int, int***> _motif_counts_map;//map to raw counts of AA occurrences
    std::map<int, int**> _peptides;

    void do_bg_counts(int* peptide, int length, int start);
    void update_bg_dist();
    void get_possible_starts(std::vector<int> & starts, int key);
    int random_choice(int num_choices, double* weights);

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
