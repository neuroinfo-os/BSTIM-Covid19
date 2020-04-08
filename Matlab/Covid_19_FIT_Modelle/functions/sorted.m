function pointer=sorted(knots, points)
% SORTED	Locate data points in knot sequence.
%
%        pointer=sorted(knots, points)
%
% given  o r d e r e d  sequences  knots  and  points,  returns the index 
% sequence  pointer  (a  r o w  vector)  for which  
%
%   pointer(j) = #{ i : knots(i)  <=  points(j) },  all  j .
%
% If the input is  n o t  o r d e r e d , the output will not mean much.

% C. de Boor / latest change: May 27, 1989
% Copyright (c) 1990-94 by Carl de Boor and The MathWorks, Inc.
% $Revision: 1.5 $  $Date: 1994/01/24 23:26:39 $

diff(sort([points(:)' knots(:)'])); tol=min(ans(find(ans>0)))/3;
ll=length(points)+length(knots);
[junk,index]=sort([knots(:)' points(:)']+[1:ll]*tol/ll);
pointer=find(index>length(knots))-[1:length(points)];
