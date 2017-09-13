#ifndef GIBBS_PY_H_
#define GIBBS_PY_H_

#define ALPHABET_LENGTH 20

#include <boost/python.hpp>
#include <map>

namespace bpy = boost::python;

namespace Gibbs{
  class Gibbs_Py{
  public:
    //keep public copies of these to pass back to python at the end so we can plot
    bpy::dict _motif_counts;
    bpy::dict _motif_start_dists;
    bpy::dict _motif_class_dists;
    bpy::list _bg_counts;
    bpy::list _tot_bg_counts;
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
	     int num_motif_classes);
//    Gibbs_Py(const Gibbs_Py& that);//disable copy constructor

    ~Gibbs_Py();

    char const* test_print();

  private:
    double* _bg_dist;//background distro
    double***  _motif_dists;//the 3D array of motif distros
    std::map<int, double**> _motif_start_dists_map;
    std::map<int, double**> _motif_class_dists_map;//the map of motif class distros
    std::map<int, int**> _peptides;

    //private because we can't call from python due to C++ data types
    double get_tot_prob(int* peptide,
			double* bg_dist,
			double*** motif_dists,
			double* class_dist,//specific distro we're using here
			double* start_dist,
			int motif_class,
			int motif_start);

  };
}

#endif//GIBS_PY_H_
