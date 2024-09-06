#include "hello_hdf5.h"
#include <H5Apublic.h>
#include <H5Gpublic.h>
#include <H5Ipublic.h>
#include <filesystem>

void add_attribute(hid_t &loc, const std::string &name, const std::string &key,
                   const std::string &value) {
  auto scalar = H5Screate(H5S_SCALAR);

  auto strtype = H5Tcopy(H5T_C_S1);
  H5Tset_size(strtype, value.size());
  H5Tset_strpad(strtype, H5T_STR_NULLTERM);

  // Delete the attribute if it has been created.
  hid_t attr = -1;
  auto attr_exist_res =
      H5Aexists_by_name(loc, name.c_str(), key.c_str(), H5P_DEFAULT);
  if (attr_exist_res > 0) {
    H5Adelete_by_name(loc, name.c_str(), key.c_str(), H5P_DEFAULT);
  }
  attr = H5Acreate_by_name(loc, (name.size() >= 1) ? name.c_str() : ".",
                           key.c_str(), strtype, scalar, H5P_DEFAULT,
                           H5P_DEFAULT, H5P_DEFAULT);

  H5Awrite(attr, strtype, value.c_str());
  H5Aclose(attr);

  H5Tclose(strtype);
  H5Sclose(scalar);
}

void add_attribute(hid_t &loc, const std::string &name, const std::string &key,
                   const double value) {
  auto scalar = H5Screate(H5S_SCALAR);

  // Create or open an attribute.
  hid_t attr = -1;
  auto attr_exist_res =
      H5Aexists_by_name(loc, name.c_str(), key.c_str(), H5P_DEFAULT);
  if (attr_exist_res > 0) {
    H5Adelete_by_name(loc, name.c_str(), key.c_str(), H5P_DEFAULT);
  }
  attr = H5Acreate_by_name(loc, name.c_str(), key.c_str(), H5T_NATIVE_DOUBLE,
                           scalar, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  H5Awrite(attr, H5T_NATIVE_DOUBLE, &value);
  H5Aclose(attr);

  H5Sclose(scalar);
}

void add_attribute(hid_t &loc, const std::string &name, const std::string &key,
                   const int value) {
  auto scalar = H5Screate(H5S_SCALAR);

  // Create or open an attribute.
  hid_t attr = -1;
  auto attr_exist_res =
      H5Aexists_by_name(loc, name.c_str(), key.c_str(), H5P_DEFAULT);
  if (attr_exist_res > 0) {
    H5Adelete_by_name(loc, name.c_str(), key.c_str(), H5P_DEFAULT);
  }
  attr = H5Acreate_by_name(loc, name.c_str(), key.c_str(), H5T_NATIVE_INT,
                           scalar, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  H5Awrite(attr, H5T_NATIVE_INT, &value);
  H5Aclose(attr);

  H5Sclose(scalar);
}

bool check_path_exists(hid_t id, const std::string &path) {
  return H5Lexists(id, path.c_str(), H5P_DEFAULT) > 0;
}

hid_t get_file(const std::string &filename) {
  hid_t file_id = -1;
  if (!std::filesystem::exists(filename)) {
    file_id =
        H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  } else {
    file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  }
  return file_id;
}

hid_t get_group(hid_t loc, const std::string &name) {
  hid_t group_id = -1;
  if (!check_path_exists(loc, name)) {
    group_id =
        H5Gcreate(loc, name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  } else {
    group_id = H5Gopen(loc, name.c_str(), H5P_DEFAULT);
  }
  return group_id;
}

hid_t get_complex_dtype() {
  hid_t complex_id = H5Tcreate(H5T_COMPOUND, sizeof(hdf5_complex_t));
  H5Tinsert(complex_id, "real", HOFFSET(hdf5_complex_t, re), H5T_NATIVE_DOUBLE);
  H5Tinsert(complex_id, "imaginary", HOFFSET(hdf5_complex_t, im),
            H5T_NATIVE_DOUBLE);
  return complex_id;
}

hid_t get_dataset(hid_t group_id, const std::string &dataset_name,
                  hid_t datatype_id, hid_t dataspace_id) {
  hid_t dataset_id = -1;
  if (!check_path_exists(group_id, dataset_name)) {
    dataset_id = H5Dcreate(group_id, dataset_name.c_str(), datatype_id,
                           dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  } else {
    dataset_id = H5Dopen(group_id, dataset_name.c_str(), H5P_DEFAULT);
  }
  return dataset_id;
}