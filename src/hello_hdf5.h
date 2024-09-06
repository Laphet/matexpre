#pragma once

#include <H5Apublic.h>
#include <H5Ipublic.h>
#include <H5Ppublic.h>
#include <H5Tpublic.h>
#include <stdlib.h>
#include <string>

typedef struct {
  double re; /*real part*/
  double im; /*imaginary part*/
} hdf5_complex_t;

void add_attribute(hid_t &loc, const std::string &name, const std::string &key,
                   const std::string &value);

void add_attribute(hid_t &loc, const std::string &name, const std::string &key,
                   const double value);

void add_attribute(hid_t &loc, const std::string &name, const std::string &key,
                   const int value);

bool check_path_exists(hid_t id, const std::string &path);

hid_t get_file(const std::string &filename);

hid_t get_group(hid_t loc, const std::string &name);

hid_t get_complex_dtype();

hid_t get_dataset(hid_t group_id, const std::string &dataset_name,
                  hid_t datatype_id, hid_t dataspace_id);
