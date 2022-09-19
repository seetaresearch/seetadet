/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *     <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_EXTENSION_UTILS_DETECTION_ITERATOR_H_
#define DRAGON_EXTENSION_UTILS_DETECTION_ITERATOR_H_

#include <dragon/core/common.h>

namespace dragon {

namespace detection {

template <typename MapT>
class KeyValueMapIterator
    : public std::iterator<std::input_iterator_tag, MapT> {
 public:
  typedef KeyValueMapIterator self_type;
  typedef ptrdiff_t difference_type;
  typedef MapT value_type;
  typedef MapT& reference;

  KeyValueMapIterator(
      typename MapT::key_type* key_ptr,
      typename MapT::value_type* value_ptr)
      : key_ptr_(key_ptr), value_ptr_(value_ptr) {}

  self_type operator++(int) {
    self_type ret = *this;
    key_ptr_++;
    value_ptr_++;
    return ret;
  }

  self_type operator++() {
    key_ptr_++;
    value_ptr_++;
    return *this;
  }

  self_type operator--() {
    key_ptr_--;
    value_ptr_--;
    return *this;
  }

  self_type operator--(int) {
    self_type ret = *this;
    key_ptr_--;
    value_ptr_--;
    return ret;
  }

  reference operator*() const {
    if (map_.key_ptr != key_ptr_) {
      map_.key_ptr = key_ptr_;
      map_.value_ptr = value_ptr_;
    }
    return map_;
  }

  self_type operator+(difference_type n) const {
    return self_type(key_ptr_ + n, value_ptr_ + n);
  }

  self_type& operator+=(difference_type n) {
    key_ptr_ += n;
    value_ptr_ += n;
    return *this;
  }

  self_type operator-(difference_type n) const {
    return self_type(key_ptr_ - n, value_ptr_ - n);
  }

  self_type& operator-=(difference_type n) {
    key_ptr_ -= n;
    value_ptr_ -= n;
    return *this;
  }

  difference_type operator-(self_type other) const {
    return key_ptr_ - other.key_ptr_;
  }

  bool operator<(const self_type& rhs) const {
    return key_ptr_ < rhs.key_ptr_;
  }

  bool operator<=(const self_type& rhs) const {
    return key_ptr_ <= rhs.key_ptr_;
  }

  bool operator==(const self_type& rhs) const {
    return key_ptr_ == rhs.key_ptr_;
  }

  bool operator!=(const self_type& rhs) const {
    return key_ptr_ != rhs.key_ptr_;
  }

 private:
  mutable MapT map_;
  typename MapT::key_type* key_ptr_;
  typename MapT::value_type* value_ptr_;
};

} // namespace detection

} // namespace dragon

#endif // DRAGON_EXTENSION_UTILS_DETECTION_ITERATOR_H_
