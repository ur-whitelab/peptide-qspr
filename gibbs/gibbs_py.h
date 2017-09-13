#include <boost/python.hpp>

namespace bpy = boost::python;
namespace bnp = boost::numpy;

void convert_list(bpy::list o, double* array, unsigned int length) {
  //takes in a python list, puts it into an array
  std::size_t n = bpy::len(o);
  if(n > length)
    EDM::edm_error("Tried to convert a list that was too big", "edm_bias_py.cpp:convert_list");  
  for (int i = 0; i < n; i++) {
    array[i] = bpy::extract<double>(o[i]);
  }
}

