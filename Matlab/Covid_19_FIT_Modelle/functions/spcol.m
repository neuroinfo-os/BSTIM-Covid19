function colloc=spcol(knots,k,tau,slvblk)
 
%       colloc=spcol(knots,k,tau[,slvblk])
%
%  returns the matrix  colloc = (D^m(i)B_j(tau_i)) , with  
%   B_j  the j-th B-spline of order  k  for the knot sequence  knots ,
%   tau  a sequence of points, assumed to be  i n c r e a s i n g , and  
%   m(i) := #{j<i: tau_j = tau_i}.
%
%     If the fourth argument is present, the matrix is returned in the almost
%  block-diagonal format (specialised for splines) required by  slvblk.m .
%
%     The recurrence relations are used to generate the entries of the matrix.

% Carl de Boor / latest change: May 13, 1989
% Carl de Boor / latest change: 19 apr 93; correct output for case  n < 1
% Copyright (c) 1990-94 by Carl de Boor and The MathWorks, Inc.
% $Revision: 1.6 $  $Date: 1994/01/24 23:26:42 $

if (~isempty(find(diff(knots)<0))),
   error('The knot sequence should be nondecreasing!')
end

%  compute the number  n  of B-splines of order  k  supported by the given
%  knot sequence and return empty matrix in case there aren't any.

npk=length(knots);n=npk-k;
if (n<1), fprintf('there are no B-splines for the given input to spcol\n')
   colloc = []; return
end

% remove all multiplicities from  tau .
nrows=length(tau);
index=[1 find(diff(tau)>0)+1];m=diff([index nrows+1]);pts=tau(index);

%set some abbreviations
nd=min([k,max(m)]);   km1=k-1;

%  augment knot sequence to provide a k-fold knot at each end, in order to avoid
% struggles near ends of interval  [knots(1),knots(npk)] . 

[augknot,addl]=augknt(knots,k);naug=length(augknot)-k;
pts=pts(:);augknot=augknot(:);

%  For each  i , determine  savl(i)  so that  k <= savl(i) < naug+1 , and,
% within that restriction,
%        augknot(savl(i)) <= pts(i) < augknot(savl(i)+1) .

savl=max([sorted(augknot(1:naug),pts);k*ones(1,length(pts))]);

b=zeros(nrows,k);

% first do those without derivatives
index1=find(m==1);
if(~isempty(index1)),
   pt1s=pts(index1);savls=savl(index1);lpt1=length(index1);
   % initialize the  b  array.
   lpt1s=index(index1); b(lpt1s,1)=ones(lpt1,1);
   
   % run the recurrence simultaneously for all  pt1(i) .
   for j=1:km1;
      saved=zeros(lpt1,1);
      for r=1:j;
         tr=augknot(savls+r)-pt1s;
         tl=pt1s-augknot(savls+r-j);
         term=b(lpt1s,r)./(tr+tl);
         b(lpt1s,r)=saved+tr.*term;
         saved=tl.*term;
      end
      b(lpt1s,j+1)=saved;
   end
end

% then do those with derivatives, if any:
if (nd>1),
   indexm=find(m>1);ptss=pts(indexm);savls=savl(indexm);lpts=length(indexm);
   % initialize the  bb  array.
   bb=ones(nd*lpts,1)*[1 zeros(1,km1)];
   lptss=nd*[1:lpts];
   
   % run the recurrence simultaneously for all  pts(i) .
   % First, bring it up to the intended level:
   for j=1:k-nd;
      saved=zeros(lpts,1);
      for r=1:j;
         tr=augknot(savls+r)-ptss;
         tl=ptss-augknot(savls+r-j);
         term=bb(lptss,r)./(tr+tl);
         bb(lptss,r)=saved+tr.*term;
         saved=tl.*term;
      end
      bb(lptss,j+1)=saved;
   end

   % save the B-spline values in successive blocks in  bb .
   
   for jj=1:nd-1;
      j=k-nd+jj; saved=zeros(lpts,1); lptsn=lptss-1;
      for r=1:j;
         tr=augknot(savls+r)-ptss;
         tl=ptss-augknot(savls+r-j);
         term=bb(lptss,r)./(tr+tl);
         bb(lptsn,r)=saved+tr.*term;
         saved=tl.*term;
      end
      bb(lptsn,j+1)=saved; lptss=lptsn;
   end

   % now use the fact that derivative values can be obtained by differencing:

   for jj=nd-1:-1:1;
      j=k-jj;
      [jj:nd-1]'*ones(1,lpts)+ones(nd-jj,1)*lptsn;lptss=ans(:); 
      for r=j:-1:1,
         ones(nd-jj,1)*(augknot(savls+r)-augknot(savls+r-j))'/j;
         bb(lptss,r)=-bb(lptss,r)./ans(:);
         bb(lptss,r+1)=bb(lptss,r+1) - bb(lptss,r);
      end
   end

   % finally, combine appropriately with  b  by interspersing the multiple 
   % point conditions appropriately: 
   dtau=diff([tau(1)-1 tau(:)' tau(nrows)+1]);
   index=find(min([dtau(2:nrows+1);dtau(1:nrows)])==0); % Determines all rows
                                                    % involving multiple tau.
   dtau=diff(tau(index));index2=find(dtau>0)+1;     % We need to make sure to
   index3=[1 (dtau==0)];                            % skip unwanted derivs:
   if (~isempty(index2)),
                      index3(index2)=1+nd-m(indexm(1:length(indexm)-1));end
   b(index,:)=bb(cumsum(index3),:);

   % and appropriately enlarge  savl
   index=cumsum([1 (diff(tau)>0)]);  
   savl=savl(index);
end

% Finally, zero out all rows of  b  corresponding to  tau  outside 
%  [knots(1),knots(npk)] .

index=find(tau<knots(1)|tau>knots(npk));
if (~isempty(index)),
   [1-nd:0]'*ones(1,length(index))+nd*ones(nd,1)*index(:)';   
   b(ans(:),:)=zeros(nd*length(index),k);
end

% The first B-spline of interest begins at knots(1), i.e., at  augknot(1+addl)
% (since  augknot's  first knot has exact multiplicity  k ). If  addl<0 ,
% this refers to a nonexistent index and means that the first  -addl  columns
% of the collocation matrix should be trivial.  This we manage by setting
savl=savl+max([0,-addl]);

if (nargin<4), % return the collocation matrix in standard matrix form
   width=max([n,naug])+km1+km1;
   cc=zeros(nrows*width,1);
   cc([1-nrows:0]'*ones(1,k)+nrows*(savl'*ones(1,k)+ones(nrows,1)*[-km1:0]))=b;
   % (This uses the fact that, for a column vector  v  and a matrix  A ,
   %  v(A)(i,j)=v(A(i,j)), all i,j.)
   colloc=zeros(nrows,n);
   colloc(:)=...
          cc([1-nrows:0]'*ones(1,n)+nrows*ones(nrows,1)*(max([0,addl])+[1:n]));

else,         % return the collocation matrix in almost block diagonal form.
              % For this, make the blocks out of the entries with the same
              %  savl(i) , with  last  computed from the differences.
   % There are two issues, the change in the B-splines considered because of
   % the use of  augknot  instead of  knots , and the possible drop of B-splines
   % because the extreme  tau  fail to involve the extreme knot intervals.

   % savl(j) is the index in  augknot  of the left knot for  tau(j) , hence the
   % corresponding row involves  B-splines to index  savl(j) wrto augknot, i.e.,
   % B-splines to index  savl(j)-addl  wrto  knots. 
   % Those with negative index are removed by cutting out their columns (i.e.,
   % shifting the blocks in which they lie appropriately). Those with index 
   % greater than  n  will be ignored because of  last .

   if (addl>0), % if B-splines were added on the left, remove them now:
      width=km1+k;cc=zeros(nrows*width,1);
      index=min([k*ones(1,length(savl));savl-addl]);
    cc([1-nrows:0]'*ones(1,k)+nrows*(index'*ones(1,k)+ones(nrows,1)*[0:km1]))=b;
      b(:)=cc([1-nrows:0]'*ones(1,k)+nrows*(ones(nrows,1)*(k+[0:km1])));
      savl=savl+k-index;
   end
   ds=(diff(savl));
   index=[0 find(ds>0) nrows];
   rows=diff(index);
   nb=length(index)-1;
   last=ds(index(2:nb));
   if(addl<0), nb=nb+1; rows=[0 rows]; last=[-addl last]; end
   addr=naug-n-addl;
   if(addr<0), nb=nb+1; rows=[rows 0]; last=[last -addr]; end
   colloc=[41 nb rows k last n-sum(last) b(:)'];
end
