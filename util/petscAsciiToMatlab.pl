#!/usr/bin/perl -w
#
#  Parallel Sweeping Preconditioner (PSP): a distributed-memory implementation
#  of a sweeping preconditioner for 3d Helmholtz equations.
#
#  Copyright (C) 2011 Jack Poulson, Lexing Ying, and
#  The University of Texas at Austin
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

# This script is for converting the PETSc ASCII sparse complex matrix output 
# files into a Matlab-readable format.

$numArgs = $#ARGV + 1;
( $numArgs >= 2 ) or die "Must provide input then output filenames.\n\n";

print "Input file is $ARGV[0]\n";
print "Output file is $ARGV[1]\n\n";

open INFILE, "<", $ARGV[0] or die $!;
open OUTFILE, ">", $ARGV[1] or die $!;

while (<INFILE>) {
  chomp($_);
  if( $_ =~ m/row (\d+):/ ){
    $_ =~ s/row (\d+):\s+//;
    $rowNumber = $1;
    @items = split /\(/, $_;
    # The first match is the empty string, so skip ahead
    for( $i=1; $i<=$#items; $i++ ){
      $item = $items[$i];
      # Try to match a full complex number with positive imaginary part
      if( $item =~ m/(\d+)\, (\-?\d+\.?\d*) \+ (\d+\.?\d*) i\)/ ){
        print OUTFILE "$rowNumber $1 $2 $3\n";
      }
      # Try to match a full complex number with negative imaginary part
      elsif( $item =~ m/(\d+)\, (\-?\d+\.?\d*) \- (\d+\.?\d*) i\)/ ){
        print OUTFILE "$rowNumber $1 $2 -$3\n";
      }
      # Try to match a real number
      elsif( $item =~ m/(\d+)\, (\-?\d+\.?\d*)\s*\)/ ){
        print OUTFILE "$rowNumber $1 $2 0\n";
      }
      # Try to match an imaginary number
      elsif( $item =~ m/(\d+)\, (\-?\d+\.?\d*) i\)/ ){
        print OUTFILE "$rowNumber $1 0 $2\n";
      }
      else{
        die "Could not match line: entry $item in line $_";
      }
    }
  }
}

close INFILE;
close OUTFILE;

