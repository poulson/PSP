#!/bin/bash
perl -pi -e 'if(/\</){s/^.*$//s}' *.vti
