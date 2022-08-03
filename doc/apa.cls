%
%
% apa.cls  version 1.02
%
% Athanassios Protopapas, July 1996
%
% Use with LaTeX2e, \documentclass[man,jou,doc]{apa}
% Default mode is doc. Conforms to the APA manual 4th ed.
% Available commands as with apaman.sty, apajou.sty, with
% the addition of \shorttitle (as per the APA requirements)
%
%%%%%%%%%%%%%%%%%
%
% Alteration log:
%
% 1.00 --> 1.01
% July 5, 1996:  \r@headl now defaults to \r@headr in [doc] (used to complain)
%
% 1.01 --> 1.02
% October 28, 1996:  Changed \rhead to \markright in [doc]
%
%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{apa}[1996/07/05 American Psychological Association format]

\DeclareOption{man}{%
\def\def@man{\@manmode}
}

\DeclareOption{jou}{%
\def\def@jou{\@joumode}
}

\DeclareOption{doc}{%
\def\def@doc{\@docmode}
}

\DeclareOption*{%
\ClassWarning{apa}{Supported options are: man, jou, doc}}

\ProcessOptions

\@ifundefined{def@man}{%
 \@ifundefined{def@jou}{%
  \def\def@doc{\@docmode}
 }{}
}{}

%=================================================================

\@ifundefined{def@man}{}{%
\LoadClass[11pt]{article}
%\RequirePackage{fancyheadings}
\RequirePackage{ulem}[1995/12/17]
%
% endnotes and endfloat loaded later!!
%
}

\@ifundefined{def@jou}{}{%
\LoadClass[twoside]{article}
\RequirePackage{twocolumn}
\RequirePackage{times}
\IfFileExists{mathptm.sty}{\RequirePackage{mathptm}}{}
}

\@ifundefined{def@doc}{}{%
\LoadClass[11pt]{article}
}

%===========================================
%
% The following is theapa.sty with minor alterations.
%
% [ theapa.sty Version 2.5.1 (May 1992)    ]
% [ by Young U. Ryu (young@emx.utexas.edu) ]
%
%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PART I: APA CITATION %%
%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%
% cite punctuations %
%%%%%%%%%%%%%%%%%%%%%

\def\BBOP{(}                % open parenthesis
\def\BBAA{\&}%               % between authors in parenthesis and ref. sec.
\def\BBAB{and}              % between authors in text
\def\BBAY{, }               % between author(s) and year
\def\BBAZ{ }                % before &
\def\BBC{; }                % between cites
\def\BBN{, }                % before note
\def\BBCP{)}                % closing parenthesis
\def\BBOQ{}                 % open quote for article title
\def\BBCQ{.}                % closing quote for article title
%
\def\BBA{\BBAA}
%
\def\citepunct#1#2#3#4#5#6#7#8#9{%
 \def\BBOP{#1}
 \def\BBAA{#2}
 \def\BBAB{#3}
 \def\BBAY{#4}
 \def\BBC{#5}
 \def\BBN{#6}
 \def\BBCP{#7}
 \def\BBOQ{#8}
 \def\BBCQ{#9}
 \def\BBA{\BBAA}}
%
\def\BBACOMMA{,}           % comma between authors in ref. sec
                           % when no. of authors = 2
\def\NoCommaBetweenTwoAuthors{\def\BBACOMMA{}}
\NoCommaBetweenTwoAuthors
%
\let\Bem\em

%%%%%%%%%%%%%%%
% cite labels %
%%%%%%%%%%%%%%%

\def\citelabels{@citelabelsA\@citelabelsB}

\def\@citelabelsA#1#2#3#4#5#6#7#8{%
     \def\BED{#1}                % editor
     \def\BEDS{#2}
     \def\BVOL{#3}
     \def\BNUM{#4}
     \def\BEd{#5}                % edition
     \def\BPG{#6}
     \def\BPGS{#7}
     \def\BCH{#8}}

\def\@citelabelsB#1#2#3#4{%
     \def\BTR{#1}
     \def\OF{#2}
     \def\IN{#3}
     \def\BBAB{#4}}

\def\BED{(Ed.)}%            % editor
\def\BEDS{(Eds.)}%          % editors
\def\BVOL{Vol.~}%           % volume name
\def\BNUM{No.~}%            % number name
\def\BEd{edition}%          % edition
\def\BPG{p.~}%              % page
\def\BPGS{pp.~}%            % pages
\def\BCH{chap.~}%           % chapter
\def\BTR{Tech.\ rep.}%      % (default) technical report type name
\def\OF{of}%
\def\IN{In}%
\def\BBAB{and}%  %%% mwh change

\NoCommaBetweenTwoAuthors

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Upper or Lowercase article titles %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% \BUcase is dropped in theapa.bst Version 2.5.
% The following three lines are not required in Version 2.5.
% They are here temporally to maintain some compatibility with
% Version 2.4; but will be removed in the next revision.
\def\@Ucase#1{\uppercase{#1}}
\def\BUcase#1{\lowercase{\@Ucase #1}}
\def\NoTitleCaseChange{\def\BUcase##1{##1}}

%%%%%%%%%%%%%
% citations %
%%%%%%%%%%%%%

\def\@lbibitem[#1]#2{\def\BBA{\BBAA}\item[\@biblabel{#1}]\if@filesw
      { \def\BBA{\string\BBA}%
        \def\protect##1{\string ##1\space}\immediate
        \write\@auxout{\string\bibcite{#2}{#1}}%
        \def\BBA{\BBAA}}\fi\ignorespaces}

% Year is separately handled by \Y<citation key>
\def\bibcite#1#2{\global\expandafter\def\csname b@#1\endcsname{#2}% names
    \global\expandafter\def\csname Y@#1\endcsname{\@year@#2}}% year
\def\@year@\BCAY#1#2#3{#3}

% first author of a multi-author reference?
\newif\if@F@cite\@F@citetrue

%
% \cite<prefix>[postfix]{keys}
% \fullcite<prefix>[postfix]{keys}
% \shortcite<prefix>[postfix]{keys}
%       writes \citation{keys} on .aux
%       produces (prefix Authors1, Year1; Authors2, Year2; ..., postfix)
%

\def\cite{\def\BCAY##1##2##3{\BCA{##1}{##2}}%
    \@ifnextchar<{\@cite}{\@cite<>}}

\def\fullcite{\def\BCAY##1##2##3{\BCA{##1}{##1}}%
    \@ifnextchar<{\@cite}{\@cite<>}}

\def\shortcite{\def\BCAY##1##2##3{\BCA{##2}{##2}}%
    \@ifnextchar<{\@cite}{\@cite<>}}

\def\@cite<#1>{\@ifnextchar[{\def\@BBN{\BBN}\@@cite<#1>}%
                             {\def\@BBN{\ignorespaces}\@@cite<#1>[]}}
\def\@@cite<#1>[#2]#3{%
    \if@filesw\immediate\write\@auxout{\string\citation{#3}}\fi%
    \edef\@citeP{}%
    \BBOP{\ignorespaces#1\space}%
    \@for\@citeb:=#3\do{%
    \@ifundefined{b@\@citeb}%
     {\expandafter\def\csname b@\@citeb\endcsname{?}%
      \expandafter\def\csname Y@\@citeb\endcsname{?}%
      \@warning{Citation `\@citeb' on page \thepage\space undefined}%
     }%
     {\@ifundefined{flag@\@citeb}%
      {\global\expandafter\def\csname flag@\@citeb\endcsname{DUMMY}%
       \def\BCA##1##2{##1}}%
      {\def\BCA##1##2{##2}}%
     }%
    \edef\B@my@dummy{\csname b@\@citeb\endcsname}%
    \ifx\@citeP\B@my@dummy%
        \if@F@cite\@F@citefalse\else{,\ }\fi%
        \csname Y@\@citeb\endcsname%
    \else%
        \if@F@cite\@F@citefalse\else{\BBC}\fi%
        \edef\@cite@undefined{?}%
        \ifx\@cite@undefined\B@my@dummy%
          \csname Y@\@citeb\endcsname%
        \else%
          \def\BBA{\BBAA}%
          \csname b@\@citeb\endcsname\BBAY\csname Y@\@citeb\endcsname%
        \fi%
    \fi%
    \edef\@citeP{\csname b@\@citeb\endcsname}%
    }{\@BBN #2}\BBCP%
    \@F@citetrue}

%
% \citeA[postfix]{keys}
% \fullciteA[postfix]{keys}
% \shortciteA[postfix]{keys}
%       writes \citation{keys} on .aux
%       produces Authors1 (Year1), ..., AuthorN (YearN, postfix)
%

\def\citeA{\def\BCAY##1##2##3{\BCA{##1}{##2}}%
           \@ifnextchar[{\def\@BBN{\BBN}\@citeA}%
                        {\def\@BBN{\ignorespaces}\@citeA[]}}

\def\fullciteA{\def\BCAY##1##2##3{\BCA{##1}{##1}}%
           \@ifnextchar[{\def\@BBN{\BBN}\@citeA}%
                        {\def\@BBN{\ignorespaces}\@citeA[]}}

\def\shortciteA{\def\BCAY##1##2##3{\BCA{##2}{##2}}%
           \@ifnextchar[{\def\@BBN{\BBN}\@citeA}%
                        {\def\@BBN{\ignorespaces}\@citeA[]}}

\def\@citeA[#1]#2{%
    \if@filesw\immediate\write\@auxout{\string\citation{#2}}\fi%
    \edef\@citeP{}%
    \@for\@citeb:=#2\do{%
    \@ifundefined{b@\@citeb}%
     {\expandafter\def\csname b@\@citeb\endcsname{?}%
      \expandafter\def\csname Y@\@citeb\endcsname{?}%
      \@warning{Citation `\@citeb' on page \thepage\space undefined}%
     }%
     {\@ifundefined{flag@\@citeb}%
      {\global\expandafter\def\csname flag@\@citeb\endcsname{DUMMY}%
       \def\BCA##1##2{##1}}%
      {\def\BCA##1##2{##2}}%
     }%
    \edef\B@my@dummy{\csname b@\@citeb\endcsname}%
    \ifx\@citeP\B@my@dummy%
        \if@F@cite\@F@citefalse\else{,\ }\fi%
        \csname Y@\@citeb\endcsname%
    \else%
        \if@F@cite\@F@citefalse\else{\BBCP,\ }\fi%
        \def\BBA{\BBAB}%
        \csname b@\@citeb\endcsname\ \BBOP\csname Y@\@citeb\endcsname%
    \fi%
    \edef\@citeP{\csname b@\@citeb\endcsname}%
    }{\@BBN #1}\BBCP%
    \@F@citetrue}

%
% \citeauthor{keys}
% \fullciteauthor{keys}
% \shortciteauthor{keys}
%       writes \citation{keys} on .aux
%       produces Authors1, ..., AuthorN
%

\def\citeauthor{\def\BCAY##1##2##3{\BCA{##1}{##2}}\@citeauthor}

\def\fullciteauthor{\def\BCAY##1##2##3{\BCA{##1}{##1}}\@citeauthor}

\def\shortciteauthor{\def\BCAY##1##2##3{\BCA{##2}{##2}}\@citeauthor}

\def\@citeauthor#1{%
    \if@filesw\immediate\write\@auxout{\string\citation{#1}}\fi%
    \edef\@citeP{}%
    \@for\@citeb:=#1\do{%
    \@ifundefined{b@\@citeb}%
     {\expandafter\def\csname b@\@citeb\endcsname{?}%
      \expandafter\def\csname Y@\@citeb\endcsname{?}%
      \@warning{Citation `\@citeb' on page \thepage\space undefined}%
     }%
     {\@ifundefined{flag@\@citeb}%
      {\global\expandafter\def\csname flag@\@citeb\endcsname{DUMMY}%
       \def\BCA##1##2{##1}}%
      {\def\BCA##1##2{##2}}%
     }%
    \edef\B@my@dummy{\csname b@\@citeb\endcsname}%
    \ifx\@citeP\B@my@dummy%
    \else%
        \if@F@cite\@F@citefalse\else{,\ }\fi%
        \def\BBA{\BBAB}%
        \csname b@\@citeb\endcsname%\ \BBOP\csname Y@\@citeb\endcsname%
    \fi%
    \edef\@citeP{\csname b@\@citeb\endcsname}%
    }%
    \@F@citetrue}

%
% \citeyear[postfix]{keys}
%       writes \citation{keys} on .aux
%       produces (Year1, ..., YearN, postfix)
%

\def\citeyear{\def\BCAY##1##2##3{}%
           \@ifnextchar[{\def\@BBN{\BBN}\@citeyear}%
                        {\def\@BBN{\ignorespaces}\@citeyear[]}}

\def\@citeyear[#1]#2{%
    \if@filesw\immediate\write\@auxout{\string\citation{#2}}\fi%
    \edef\@citeP{}%
    \BBOP\@for\@citeb:=#2\do{%
    \@ifundefined{b@\@citeb}%
     {\expandafter\def\csname b@\@citeb\endcsname{?}%
      \expandafter\def\csname Y@\@citeb\endcsname{?}%
      \@warning{Citation `\@citeb' on page \thepage\space undefined}%
     }%
     {\@ifundefined{flag@\@citeb}%
      {\global\expandafter\def\csname flag@\@citeb\endcsname{DUMMY}}%
      {}%
     }%
    \edef\B@my@dummy{\csname b@\@citeb\endcsname}%
    \if@F@cite\@F@citefalse\else{,\ }\fi%
    \csname Y@\@citeb\endcsname%
    \edef\@citeP{\csname b@\@citeb\endcsname}%
    }{\@BBN #1}\BBCP%
    \@F@citetrue}

\def\nocite#1{\@bsphack%
    \if@filesw\immediate\write\@auxout{\string\citation{#1}}\fi%
    \@for\@citeb:=#1\do{%
    \@ifundefined{b@\@citeb}%
    {\edef\B@my@dummy{*}%
     \ifx\@citeb\B@my@dummy\else
     \@warning{Citation `\@citeb' on page \thepage\space undefined}\fi}%
    {\global\expandafter\def\csname flag@\@citeb\endcsname{DUMMY}}}%
    \@esphack}

%
% To maintain compatibility with Version 2.4
% \citeB = \citeA
% \fullciteB = \fullciteA
% \shortciteB = \shortciteA
% \citeC = \citeauthor
% \fullciteC = \fullciteauthor
% \shortciteC = \shortciteauthor

\let\citeB=\citeA
\let\fullciteB=\fullciteA
\let\shortciteB=\shortciteA

\let\citeC=\citeauthor
\let\fullciteC=\fullciteauthor
\let\shortciteC=\shortciteauthor

%
% No labels in the bibliography.
\def\@biblabel#1{}
\newskip{\bibleftmargin}
%\bibleftmargin=2.5em
%
\def\refname{References}
\def\bibname{Bibliography}
%
\def\thebibliography#1{%
     \bibnewpage
     \@ifundefined{chapter}%
       {\section*{\refname}%
        \@mkboth{\uppercase{\refname}}{\uppercase{\refname}}
        \addcontentsline{toc}{section}{\refname}}%
       {\chapter*{\bibname}%
        \@mkboth{\uppercase{\bibname}}{\uppercase{\bibname}}
        \addcontentsline{toc}{chapter}{\bibname}}
     \bibsize
     \list
     {\relax}{\bib@indent}
     \clubpenalty4000\widowpenalty4000
     \sfcode`\.=1000\relax}
\let\endthebibliography=\endlist

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PART II: APA SECTION HEADING & SERIATION %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%
% \apa@startsection{Heading Indent}{Beforeskip}{Afterskip}{Heading}
%
\def\apa@startsection#1#2#3{\@startsection{}{}{#1}{#2}{#3}{\relax}*}

%
% \b@level@LEVEL-NO@skip
%       Skip before heading: If >= 0, the following par is NOT indented.
%                            If < 0, it is indented.
%                            \b@level@LEVEL-NO@skip = vertical skip
%
% \e@level@LEVEL-NO@skip
%       Skip after heading: If > 0, the following par starts in a new line;
%                               \e@level@LEVEL-NO@skip = vertical skip
%                           If <= 0, it starts just after the heading;
%                               \e@level@LEVEL-NO@skip = horizontal skip
%
\newskip\b@level@one@skip   \b@level@one@skip=2.5ex plus 1ex minus .2ex
\newskip\e@level@one@skip   \e@level@one@skip=1.5ex plus .6ex minus .1ex
\newskip\b@level@two@skip   \b@level@two@skip=2.5ex plus 1ex minus .2ex
\newskip\e@level@two@skip   \e@level@two@skip=1.5ex plus .6ex minus .1ex
\newskip\b@level@three@skip \b@level@three@skip=2.0ex plus .8ex minus .2ex
\newskip\e@level@three@skip \e@level@three@skip=1.5ex plus .6ex minus .1ex
\newskip\b@level@four@skip  \b@level@four@skip=1.8ex plus .8ex minus .2ex
\newskip\e@level@four@skip  \e@level@four@skip=1.5ex plus .6ex minus .1ex
\newskip\b@level@five@skip  \b@level@five@skip=1.8ex plus .8ex minus .2ex
\newskip\e@level@five@skip  \e@level@five@skip=0ex

%
% Selecting the Levels of Headings (APA, pp. 66-67)
%

% One Level
\def\OneLevelHeading{%
    \def\section{\@ifnextchar*{\@sections}%
                {\@ifnextchar[{\@sectionb}{\@section}}}%
    \def\@sections*##1{\leveltwo{##1}}%
    \def\@sectionb[##1]##2{\leveltwo{##2}%
        \@mkboth{\uppercase{##1}}{\uppercase{##1}}%
        \addcontentsline{toc}{section}{##1}}%
    \def\@section##1{\leveltwo{##1}%
        \@mkboth{\uppercase{##1}}{\uppercase{##1}}%
        \addcontentsline{toc}{section}{##1}}%
}

% Two Levels
\def\TwoLevelHeading{%
    \def\section{\@ifnextchar*{\@sections}%
                {\@ifnextchar[{\@sectionb}{\@section}}}%
    \def\@sections*##1{\leveltwo{##1}}%
    \def\@sectionb[##1]##2{\leveltwo{##2}%
        \@mkboth{\uppercase{##1}}{\uppercase{##1}}%
        \addcontentsline{toc}{section}{##1}}%
    \def\@section##1{\leveltwo{##1}%
        \@mkboth{\uppercase{##1}}{\uppercase{##1}}%
        \addcontentsline{toc}{section}{##1}}%
%
    \def\subsection{\@ifnextchar*{\@subsections}%
                   {\@ifnextchar[{\@subsectionb}{\@subsection}}}%
    \def\@subsections*##1{\levelfour{##1}}%
    \def\@subsectionb[##1]##2{\levelfour{##2}%
        \addcontentsline{toc}{subsection}{##1}}%
    \def\@subsection##1{\levelfour{##1}%
        \addcontentsline{toc}{subsection}{##1}}%
}

% Two Levels - Variation
\def\TwoLevelHeadingA{%
    \def\section{\@ifnextchar*{\@sections}%
                {\@ifnextchar[{\@sectionb}{\@section}}}%
    \def\@sections*##1{\leveltwo{##1}}%
    \def\@sectionb[##1]##2{\leveltwo{##2}%
        \@mkboth{\uppercase{##1}}{\uppercase{##1}}%
        \addcontentsline{toc}{section}{##1}}%
    \def\@section##1{\leveltwo{##1}%
        \@mkboth{\uppercase{##1}}{\uppercase{##1}}%
        \addcontentsline{toc}{section}{##1}}%
%
    \let\subsection=\levelfive%
}

% Three Levels
\def\ThreeLevelHeading{%
    \def\section{\@ifnextchar*{\@sections}%
                {\@ifnextchar[{\@sectionb}{\@section}}}%
    \def\@sections*##1{\leveltwo{##1}}%
    \def\@sectionb[##1]##2{\leveltwo{##2}%
        \@mkboth{\uppercase{##1}}{\uppercase{##1}}%
        \addcontentsline{toc}{section}{##1}}%
    \def\@section##1{\leveltwo{##1}%
        \@mkboth{\uppercase{##1}}{\uppercase{##1}}%
        \addcontentsline{toc}{section}{##1}}%
%
    \def\subsection{\@ifnextchar*{\@subsections}%
                   {\@ifnextchar[{\@subsectionb}{\@subsection}}}%
    \def\@subsections*##1{\levelfour{##1}}%
    \def\@subsectionb[##1]##2{\levelfour{##2}%
        \addcontentsline{toc}{subsection}{##1}}%
    \def\@subsection##1{\levelfour{##1}%
        \addcontentsline{toc}{subsection}{##1}}%
%
    \let\subsubsection=\levelfive
% If you want subsubsection headings listed in the table of contents
% remove the above \let\subsub... and uncomment the following.
%    \def\subsubsection{\@ifnextchar*{\@subsubsections}%
%                      {\@ifnextchar[{\@subsubsectionb}{\@subsubsection}}}%
%    \def\@subsubsections*##1{\levelfive{##1}}%
%    \def\@subsubsectionb[##1]##2{\levelfive{##2}%
%        \addcontentsline{toc}{subsubsection}{##1}}%
%    \def\@subsubsection##1{\levelfive{##1}%
%        \addcontentsline{toc}{subsubsection}{##1}}%
}

% Four Levels
\def\FourLevelHeading{%
    \def\section{\@ifnextchar*{\@sections}%
                {\@ifnextchar[{\@sectionb}{\@section}}}%
    \def\@sections*##1{\leveltwo{##1}}%
    \def\@sectionb[##1]##2{\leveltwo{##2}%
        \@mkboth{\uppercase{##1}}{\uppercase{##1}}%
        \addcontentsline{toc}{section}{##1}}%
    \def\@section##1{\leveltwo{##1}%
        \@mkboth{\uppercase{##1}}{\uppercase{##1}}%
        \addcontentsline{toc}{section}{##1}}%
%
    \def\subsection{\@ifnextchar*{\@subsections}%
                   {\@ifnextchar[{\@subsectionb}{\@subsection}}}%
    \def\@subsections*##1{\levelthree{##1}}%
    \def\@subsectionb[##1]##2{\levelthree{##2}%
        \addcontentsline{toc}{subsection}{##1}}%
    \def\@subsection##1{\levelthree{##1}%
        \addcontentsline{toc}{subsection}{##1}}%
%
    \def\subsubsection{\@ifnextchar*{\@subsubsections}%
                      {\@ifnextchar[{\@subsubsectionb}{\@subsubsection}}}%
    \def\@subsubsections*##1{\levelfour{##1}}%
    \def\@subsubsectionb[##1]##2{\levelfour{##2}%
        \addcontentsline{toc}{subsubsection}{##1}}%
    \def\@subsubsection##1{\levelfour{##1}%
        \addcontentsline{toc}{subsubsection}{##1}}%
%
    \let\paragraph=\levelfive%
}

% Five Levels
\def\FiveLevelHeading{%
    \def\section{\@ifnextchar*{\@sections}%
                {\@ifnextchar[{\@sectionb}{\@section}}}%
    \def\@sections*##1{\levelone{##1}}%
    \def\@sectionb[##1]##2{\levelone{##2}%
        \@mkboth{\uppercase{##1}}{\uppercase{##1}}%
        \addcontentsline{toc}{section}{##1}}%
    \def\@section##1{\levelone{##1}%
        \@mkboth{\uppercase{##1}}{\uppercase{##1}}%
        \addcontentsline{toc}{section}{##1}}%
%
    \def\subsection{\@ifnextchar*{\@subsections}%
                   {\@ifnextchar[{\@subsectionb}{\@subsection}}}%
    \def\@subsections*##1{\leveltwo{##1}}%
    \def\@subsectionb[##1]##2{\leveltwo{##2}%
        \addcontentsline{toc}{subsection}{##1}}%
    \def\@subsection##1{\leveltwo{##1}%
        \addcontentsline{toc}{subsection}{##1}}%
%
    \def\subsubsection{\@ifnextchar*{\@subsubsections}%
                      {\@ifnextchar[{\@subsubsectionb}{\@subsubsection}}}%
    \def\@subsubsections*##1{\levelthree{##1}}%
    \def\@subsubsectionb[##1]##2{\levelthree{##2}%
        \addcontentsline{toc}{subsubsection}{##1}}%
    \def\@subsubsection##1{\levelthree{##1}%
        \addcontentsline{toc}{subsubsection}{##1}}%
%
    \let\paragraph=\levelfour%
    \let\subparagraph=\levelfive%
}

%
% Seriations (pp. 66-68): ONE level of seriation only.
%  \begin{seriate} \item ... \end{seriate}
%           within a paragraph or sentence
%  \begin{APAenumerate} \item ... \end{APAenumerate}
%           at the start of each paragraph in a series
%           with enumeration
%  \begin{APAitemize} \item ... \end{APAitemize}
%           at the start of each paragraph in a series
%           without enumeration
%

% Refer to LaTeX book to modify, if you want, spaces before and after of
%  \begin{...} ... \end{...} or spaces between \item-s.
\newcounter{APAenum}
\newskip\@text@par@indent
\def\APAenumerate{\@text@par@indent\parindent\setbox0\hbox{1. }%
    \list{\arabic{APAenum}.}{\usecounter{APAenum}
    \labelwidth\z@\labelsep\z@\leftmargin\z@\parsep\z@
    \rightmargin\z@\itemsep\z@\topsep\z@\partopsep\z@
    \itemindent\@text@par@indent\advance\itemindent by\wd0
    \def\makelabel##1{\hss\llap{##1 }}}}
\let\endAPAenumerate=\endlist

\def\seriate{\@bsphack\begingroup%
   \setcounter{APAenum}{0}%
   \def\item{\addtocounter{APAenum}{1}(\alph{APAenum})\space}%
   \ignorespaces}
\def\endseriate{\endgroup\@esphack}

\def\APAitemize{\@text@par@indent\parindent\setbox0\hbox{$\bullet$}%
    \list{$\bullet$}{%
    \labelwidth\z@\labelsep.5em\leftmargin\z@\parsep\z@
    \rightmargin\z@\itemsep\z@\topsep\z@\partopsep\z@
    \itemindent\@text@par@indent
    \advance\itemindent by\wd0\advance\itemindent by.5em
    \def\makelabel##1{\hss\llap{##1}}}}
\let\endAPAitemize=\endlist

\let\enumerate=\APAenumerate
\let\endenumerate=\endAPAenumerate
\let\itemize=\APAitemize
\let\enditemize=\endAPAitemize

%=============================================================
%
% End of theapa.sty
%
%=============================================================


%%%%%%%%%%%%%%%%%%%%%%%
%                     %
%  MANUSCRIPT FORMAT  %
%                     %
%%%%%%%%%%%%%%%%%%%%%%%

\@ifundefined{def@man}{}{%

% Level One:   Centered Uppercase Heading
% Level Two:   Centered Boldface Heading
% Level Three: Centered Underlined Heading
% Level Four:  Flushleft Underlined Heading occuping a whole line
% Level Five:  Intended Underlined Heading followed by texts in the same line
%
\def\levelone#1{\apa@startsection%
    {\z@}{\b@level@one@skip}{\e@level@one@skip}%
    {\leftskip4em plus 1fill\rightskip\leftskip\parindent\z@\relax%
     \uppercase\expandafter{#1}}}    % <- \bf
\def\leveltwo#1{\apa@startsection%
    {\z@}{\b@level@two@skip}{\e@level@two@skip}%
    {\leftskip4em plus 1fill\rightskip\leftskip\parindent\z@\relax%
     {\bf #1}}}                           % <- \bf
\def\levelthree#1{\apa@startsection%
    {\z@}{\b@level@three@skip}{\e@level@three@skip}%
    {\leftskip4em plus 1fill\rightskip\leftskip\parindent\z@\relax%
    {\em #1}}}
\def\levelfour#1{\apa@startsection%
    {\z@}{\b@level@four@skip}{\e@level@four@skip}{{\em #1}}}
\def\levelfive#1{\apa@startsection%
    {\parindent}{\b@level@five@skip}{\e@level@five@skip}{{\em #1}.}\hskip .5em plus 1pt minus .5pt}

\long\def\title#1{\long\def\@title{#1}}
\long\def\author#1{\long\def\@author{#1}}
\long\def\twoauthors#1#2{\long\def\@authorOne{#1}\long\def\@authorTwo{#2}%
\long\def\@author{#1}}
\long\def\affiliation#1{\long\def\@affil{#1}}
\long\def\twoaffiliations#1#2{\long\def\@affilOne{#1}\long\def\@affilTwo{#2}%
\long\def\@affil{#1}}
\long\def\note#1{\long\def\@note{#1}}
\long\def\abstract#1{\long\def\@abstract{#1}}
\long\def\acknowledgements#1{\long\def\@acks{#1}}
\def\rightheader#1{\def\r@headr{\protect\uppercase{#1}}}
\def\leftheader#1{\def\r@headl{#1}}
\def\shorttitle#1{\def\s@title{#1}}%\markright{\hfill\rm #1 \thepage}
\def\rheadname{Running Head:}
\def\acksname{Author Note}

\def\@@spacing{1.75}

\RequirePackage{endnotes}

%========== Alterations to endnotes ===================

%\def\theendnote{\arabic{endnote}}
\def\@makeenmark{\hbox{$^{\mbox{\small\@ifundefined{hrm}{}{\sf}\@theenmark}}$}}
\def\enoteformat{\rightskip\z@ \leftskip\z@ \parindent=0.4in
     \leavevmode\llap{\hbox{$^{\mbox{\small\@ifundefined{hrm}{}{\sf}\@theenmark}}$}}}
\def\notesname{Footnotes}
\AtEndDocument{
\@ifundefined{@acks}{}{\newpage \section{\acksname} \@acks}
\@ifundefined{hrm}{\def\enotesize{\normalsize}}{\def\enotesize{\normalsize\@ifundefined{hrm}{}{\sf}}}
\raggedright\fussy\hyphenpenalty 10000
\@ifundefined{NoEndnotes}{\newpage \theendnotes}{}
%\setlength{\parindent}{0in}
}

\RequirePackage[notablist,figlist,notabhead,nofighead,tablesfirst]{endfloat}[1995/10/11]

%========= Alterations to endfloat ====================

\renewcommand{\figureplace}{%
   \par\begin{center}
   \parbox{\textwidth}
   {\begin{center}
   \rule{2.1in}{0.005in}\\
   Insert \figurename~\thepostfig\ about here\\
   \rule{2.1in}{0.005in}\\
   \end{center}}
   \end{center}\par}
\renewcommand{\tableplace}{%
   \par\begin{center}
   \parbox{\textwidth}
   {\begin{center}
   \rule{2.1in}{0.005in}\\
   Insert \tablename~\theposttbl\ about here\\
   \rule{2.1in}{0.005in}\\
   \end{center}}
   \end{center}\par}

\def\@gobbleuntilnext[#1]{}
\let\eatarg\@gobbleuntilnext
\let\ifnextchar\@ifnextchar

\def\figure{%
     \efloat@condopen{fff}
     \efloat@iwrite{fff}{\string\begin{figure*}[h]}%
     \ifnextchar[{\gobbleuntilnext[}{}
     \efloat@iwrite{fff}{\string\ifnextchar[{\string\eatarg}{}}
    \if@domarkers%
       \addtocounter{postfig}{1}% % bj
       \figureplace%              % bj
    \fi%
    \def\@currenvir{efloat@float}%
    \begingroup%
    \let\do\ef@makeinnocent \dospecials%
    \ef@makeinnocent\^^L% and whatever other special cases
    \endlinechar`\^^M \catcode`\^^M=12 \ef@xfigure}%

\def\table{\efloat@condopen{ttt}
    \efloat@iwrite{ttt}{\string\begin{table*}[h]}%
    \ifnextchar[{\gobbleuntilnext[}{}
    \@ifundefined{hrm}{}{%
    \efloat@iwrite{ttt}{\string\sf}}%
    \efloat@iwrite{ttt}{\string\ifnextchar[{\string\eatarg}{}} % bj
    \if@domarkers
       \addtocounter{posttbl}{1} % bj
       \tableplace               % bj
    \fi
    \def\@currenvir{efloat@float}%
    \begingroup
    \let\do\ef@makeinnocent \dospecials
    \ef@makeinnocent\^^L% and whatever other special cases
    \endlinechar`\^^M \catcode`\^^M=12 \ef@xtable}

\long\def\@@contentsline#1#2#3{ #2 }
\long\def\@@numberline#1#2{ \underline{Figure #1}. #2 \vspace{0.5 \baselineskip} \par }
\long\def\@@caption{\refstepcounter\@captype \@dblarg{\@@@caption\@captype}}
\long\def\@@@caption#1[#2]#3{\addcontentsline{\csname
  ext@#1\endcsname}{#1}{\protect\numberline{\csname 
  the#1\endcsname}{\ignorespaces #2}}}
\def\listoffigures{\section*{Figure Captions}\@starttoc{lof}}

\long\def\figurecaptionsformat{
  \let\contentsline=\@@contentsline
  \let\numberline=\@@numberline
  \def\baselinestretch{\@@spacing}
}

\def\processfigures{%
 \expandafter\ifnum \csname @ef@fffopen\endcsname>0
  \immediate\closeout\efloat@postfff \ef@setct{fff}{0}
  \clearpage                                                        % bj
  \if@figlist                                                       % bj
    \figurecaptionsformat
    {\normalsize\@ifundefined{hrm}{}{\sf}\listoffigures}                                     % bj
    \clearpage                                                      % bj
  \fi
  \if@fighead
     \section*{\figuresection}                                   % bj
     \suppressfloats[t]                                          % jpg
  \fi
  \markboth{}{}
  \pagestyle{empty}
  \let\caption=\@@caption
  \def\baselinestretch{\@@spacing}\normalsize\@ifundefined{hrm}{}{\sf}
  \processfigures@hook \@input{\jobname.fff}
 \fi}

\def\processtables{%
  \expandafter\ifnum \csname @ef@tttopen\endcsname>0
  \immediate\closeout\efloat@postttt \ef@setct{ttt}{0}
  \clearpage                                                       % bj
%  \if@tablist                                                     % bj TP
%    {\normalsize\listoftables}                                    % bj TP
%    \clearpage                                                    % bj TP
% \fi
  \if@tabhead
      \section*{\tablesection}                                  % bj
      \suppressfloats[t]                                        % jpg
  \fi
% \markboth{\uppercase{\tablesection}}{\uppercase{\tablesection}}% bj
  \def\baselinestretch{\@@spacing}
  \processtables@hook \@ifundefined{hrm}{}{\sf}\tiny\normalsize\@input{\jobname.ttt}
 \fi}

%========= Maketitle, margins, lengths, etc. ==========

\def\maketitle{
\@ifundefined{hrm}{}{\hrm}
 \@ifundefined{@author}{\typeout{Warning: Author not defined} \def\@author{Author}}{}
 \@ifundefined{@title}{\typeout{Warning: Title not defined} \def\@title{Title}}{}
 \@ifundefined{@affil}{\typeout{Warning: Affiliation not defined} \def\@affil{Affiliation}}{}
 \@ifundefined{@abstract}{\typeout{Warning: Abstract not defined \def\abstract{}}}{}
 \@ifundefined{s@title}{\typeout{Using title for short title} 
                        \def\s@title{\@title}
                        \markright{\rm\hfill\@title}} %\protect\thepage}}
                       {\markright{\rm\hfill\s@title}} %\protect\thepage}}
 \@ifundefined{r@headr}{\typeout{Using short title for running head}
                        \def\r@headr{\protect\uppercase{\s@title}}}{}
  \begin{center}
  \vspace*{0.5in}
  \rheadname\hspace{0.1in}\uppercase{\r@headr}\\

  \vspace*{1in}
  \@title\vspace{0.5in}\\

  \@ifundefined{@authorTwo}{
  \@author \\

  \@affil \vspace{0.25in} \\ }{
  \@authorOne \\

  \@affilOne \vspace{0.2in} \\
  \@authorTwo \\

  \@affilTwo \vspace{0.25in} \\ }

  \@ifundefined{@note}
   {\vspace*{\baselineskip} }
   {\@note}

  \end{center}
  \newpage
  \raggedright
  \hyphenpenalty 10000
  \fussy
  \section{Abstract}

  \@abstract

  \newpage
  \setlength{\parindent}{0.4in}
  \@ifundefined{DoNotRepeatTitle}{\section{\protect\@title}}{}

  \indent
}

\ThreeLevelHeading
\def\baselinestretch{\@@spacing}
\def\refname{References}
\setlength{\footnotesep}{0.5cm}
\setlength{\topmargin}{0in}
\setlength{\oddsidemargin}{0.25in}
\setlength{\evensidemargin}{0.25in}
\setlength{\textwidth}{6in}
\setlength{\textheight}{8.5in}
\setcounter{secnumdepth}{0}

\def\ps@myheadings{
 \let\@mkboth\@gobbletwo
 \def\@oddhead{\hbox{}\rightmark\hfil\hspace{0.1in}\rm\thepage}
 \def\@oddfoot{}
 \def\@evenhead{}
 \def\@evenfoot{}
 \def\sectionmark##1{}
 \def\subsectionmark##1{}}

\pagestyle{myheadings}
\raggedright
\hyphenpenalty 10000
\newcommand{\footmark}[1]{${}^{\mbox{\normalsize #1}}$}

\long\def\@makecaption#1#2{
 \vskip 10pt 
 \setbox\@tempboxa\hbox{\parbox{\columnwidth}{#1 \cap@line {\cap@style #2} }}
 \ifdim \wd\@tempboxa >\hsize \parbox{\columnwidth}{#1 \cap@line {\cap@style #2} } \par 
 \else \hbox to\hsize{\box\@tempboxa\hfil} \par
 \fi 
 \vskip \baselineskip }

\long\def\@caption#1[#2]#3{\par\addcontentsline{\csname
  ext@#1\endcsname}{#1}{\protect\numberline{\csname 
  the#1\endcsname}{\ignorespaces #2}}\begingroup
    \@parboxrestore
    \normalsize\@ifundefined{hrm}{}{\hrm}
    \@makecaption{\csname fnum@#1\endcsname}{\ignorespaces #3 } \par
  \endgroup}

\def\caption{\refstepcounter\@captype \@dblarg{\@caption\@captype}}

\def\centeredcaption#1{\caption}

\def\fnum@figure{\def\cap@style{} \def\cap@line{} {\em Figure {\thefigure}}. }
\long\def\fnum@table{\def\cap@style{\em} \def\cap@line{\\ } Table \thetable }

\setcounter{topnumber}{1}
\def\topfraction{.7}
\setcounter{bottomnumber}{1}
\def\bottomfraction{.6}
\setcounter{totalnumber}{1}
\def\textfraction{.2}
\def\floatpagefraction{.7}
\setcounter{dbltopnumber}{1}
\def\dbltopfraction{.7}
\def\dblfloatpagefraction{.7}
\def\dbltextfloatsep{\textfloatsep}

\fussy
\setlength{\parindent}{0.4in}
\let\footnote=\endnote
\def\footnotesize{\normalsize\rm}

\def\helvetica{%
 \newfont{\helvslan}{phvro at 11pt}
 \newfont{\helvbold}{phvb at 11pt}
 \newfont{\helvromn}{phvr at 11pt}
 \def\hsl{\helvslan}
 \def\hbf{\helvbold}
 \def\hrm{\helvromn}
 \RequirePackage{helvet}
 \def\bf{\hbf}
%\def\sf{\hrm}
 \def\rm{\sf}
 \def\sl{\hsl}
 \doublehyphendemerits5000
 \hfuzz0pt
 \rightskip=0pt plus .6em  % ragged-right.  Omit this line for flush-right
 \tolerance=9999
 \pretolerance=-1 % Always look for hyphens
 \emergencystretch=25pt
 \hbadness=30000
 \hyphenpenalty=100 % since average badness will be high!
 \sf
}

\def\timesroman{%
 \RequirePackage{times}
 \InputIfFileExists{mathptm.sty}{}{}
 \doublehyphendemerits5000
 \hfuzz0pt
 \tolerance=9999
 \pretolerance=-1 % Always look for hyphens
 \emergencystretch=25pt
 \hbadness=30000
 \hyphenpenalty=100 % since average badness will be high!
}

\bibleftmargin=\parindent
\def\bibnewpage{\newpage}
\def\bibsize{}
\def\bib@indent{\labelsep=0em
                \leftmargin=0em%\bibleftmargin
                \itemindent=\bibleftmargin}
                \def\newblock{\hskip .11em plus .33em minus .07em}\fussy
}

%%%%%%%%%%%%%%%%%%
%                %
% JOURNAL FORMAT %
%                %
%%%%%%%%%%%%%%%%%%

\@ifundefined{def@jou}{}{%

% Level One:   Centered Uppercase Heading
% Level Two:   Centered Heading
% Level Three: Centered Italics Heading
% Level Four:  Flushleft Italics Heading occuping a whole line
% Level Five:  Intended Italics Heading followed by texts in the same line
%
\def\levelone#1{\apa@startsection%
    {\z@}{\b@level@one@skip}{\e@level@one@skip}%
    {\leftskip4em plus 1fill\rightskip\leftskip\parindent\z@\relax%
     \uppercase\expandafter{\large #1}}}    % <- \bf
\def\leveltwo#1{\apa@startsection%
    {\z@}{\b@level@two@skip}{\e@level@two@skip}%
    {\leftskip4em plus 1fill\rightskip\leftskip\parindent\z@\relax%
     {\large #1}}}                           % <- \bf
\def\levelthree#1{\apa@startsection%
    {\z@}{\b@level@three@skip}{\e@level@three@skip}%
    {\leftskip4em plus 1fill\rightskip\leftskip\parindent\z@\relax%
    {\large{\em #1}}}}
\def\levelfour#1{\apa@startsection%
    {\z@}{\b@level@four@skip}{\e@level@four@skip}{\large{\em #1}}}
\def\levelfive#1{\apa@startsection%
    {\parindent}{\b@level@five@skip}{\e@level@five@skip}{{\em #1}.}\hskip .5em plus 1pt minus .5pt}

\long\def\title#1{\long\def\@title{#1}}
\long\def\author#1{\long\def\@author{#1}}
\long\def\twoauthors#1#2{\long\def\@authorOne{#1}\long\def\@authorTwo{#2}%
\long\def\@author{#1}}
\long\def\affiliation#1{\long\def\@affil{#1}}
\long\def\twoaffiliations#1#2{\long\def\@affilOne{#1}\long\def\@affilTwo{#2}
\long\def\@affil{#1}}
\long\def\note#1{\long\def\@note{#1}}
\long\def\abstract#1{\long\def\@abstract{#1}}
\long\def\acknowledgements#1{\long\def\@acks{#1}}
\def\rightheader#1{\def\r@headr{\protect\uppercase{\protect\small #1}}}
\def\leftheader#1{\def\r@headl{\protect\uppercase{\protect\small #1}}}
\def\shorttitle#1{}%\def\s@title{#1}}%\markright{\hfill\rm #1 \thepage}
\def\ccopyrright{}

\def\maketitle{
 \@ifundefined{@author}{\typeout{Warning: Author not defined} \def\@author{Author}}{}
 \@ifundefined{@title}{\typeout{Warning: Title not defined} \def\@title{Title}}{}
 \@ifundefined{@abstract}{\typeout{Warning: Abstract not defined}}{}
 \@ifundefined{r@headr}{\def\r@headr{\protect\uppercase{\protect\small \@title}}}{}
 \@ifundefined{r@headl}{\def\r@headl{\protect\uppercase{\protect\small \@author}}}{}
 \twocolumn[
  \begin{center}
  \ccopyrright
  {\LARGE \@title}\\
  \vspace{0.2in}
  \@ifundefined{@authorTwo} {
  {\Large \@author \vspace{0in} }
  \@ifundefined{@affil}
   {\vspace{0in}}
   {\\ \@affil}}            {
  {\Large \@authorOne \vspace{0in} }
  \@ifundefined{@affilOne}
   {\vspace{0in}}
   {\\ \@affilOne \vspace{0.05in} \\}
  {\Large \@authorTwo \vspace{0in} }
  \@ifundefined{@affilTwo}
   {\vspace{0in}}
   {\\ \@affilTwo}}
  \@ifundefined{@note}
   {\vspace{0.2in} }
   {\vspace{0.05in} \\ {\large \@note \vspace{0.2in} }}
  \@ifundefined{@abstract}
  {\par }
  {\par \parbox{5in}
   {\small \noindent \@abstract
   }\vspace{0.11in}
  }
  \end{center}
 ]
 \pagenumbering{arabic}
 \thispagestyle{empty}
 \@ifundefined{@acks}
  {}
  {\begin{figure}[b]
   \parbox{\columnwidth}{ \setlength{\parindent}{0.2in}
   \noindent \makebox[2in]{\hrulefill}

   \vspace*{0.05in}
   {\footnotesize
   \indent \@acks

   }}
   \end{figure}}
 \markboth{\hfill \r@headl \hfill}{\hfill \r@headr \hfill}
 \noindent
}

\ThreeLevelHeading
\def\baselinestretch{1}
\def\refname{References}
\setlength{\footnotesep}{0.44cm}
\setlength{\topmargin}{-0.575in}
\setlength{\oddsidemargin}{-0.1in}
\setlength{\evensidemargin}{-0.15in}
\setlength{\textwidth}{6.8in}
\setlength{\textheight}{9.15in}
\setcounter{secnumdepth}{0}
\setlength{\textfloatsep}{0.2in}
\def\ps@myheadings{\let\@mkboth\@gobbletwo
\def\@oddhead{\hbox{}\rightmark \hfil
\large\rm\thepage}\def\@oddfoot{}\def\@evenhead{\large\rm \thepage\hfil\leftmark\hbox
{}}\def\@evenfoot{}\def\sectionmark##1{}\def\subsectionmark##1{}}
\pagestyle{myheadings}

\long\def\@makecaption#1#2{
 \vskip 10pt 
 \setbox\@tempboxa\hbox{#1 #2}
 \ifdim \wd\@tempboxa >\hsize #1 #2\par \else \hbox
to\hsize{\hfil\box\@tempboxa\hfil} 
 \fi 
 \vskip 4pt}

\long\def\@makeccaption#1#2#3{
 \vskip 10pt 
 \setbox\@tempboxa\hbox{#1 #2}
 \ifdim \wd\@tempboxa > \hsize 
    \hbox to\hsize{\hfil\parbox[t]{#3}{#1 #2}\hfil} 
  \else 
    \ifdim \wd\@tempboxa > #3
      \hbox to\hsize{\hfil\parbox[t]{#3}{#1 #2}\hfil}
    \else
      \hbox to\hsize{\hfil\mbox{#1 #2}\hfil} 
    \fi
 \fi 
 \vskip 4pt}

\long\def\@caption#1[#2]#3{\par\addcontentsline{\csname
  ext@#1\endcsname}{#1}{\protect\numberline{\csname 
  the#1\endcsname}{\ignorespaces #2}}\begingroup
    \@parboxrestore
    \normalsize
    \@makecaption{\csname fnum@#1\endcsname}{\ignorespaces #3}\par
  \endgroup}

\long\def\@ccaption#1[#2]#3#4{\def\@ccwidth{#3} \par\addcontentsline{\csname
  ext@#1\endcsname}{#1}{\protect\numberline{\csname 
  the#1\endcsname}{\ignorespaces #2}}\begingroup
    \@parboxrestore
    \normalsize
    \@makeccaption{\csname fnum@#1\endcsname}{\ignorespaces #4}{\@ccwidth}\par
  \endgroup}

\def\caption{\refstepcounter\@captype \@dblarg{\@caption\@captype}}

\long\def\centeredcaption{\refstepcounter\@captype \@dblarg{\@ccaption\@captype}}

\def\fnum@figure{\small {\em Figure {\thefigure}}.\hspace{0.07in}}
\long\def\fnum@table{ \makebox[\columnwidth][l]{Table \thetable} \\ \em }

\setcounter{topnumber}{2}
\def\topfraction{.7}
\setcounter{bottomnumber}{2}
\def\bottomfraction{.6}
\setcounter{totalnumber}{3}
\def\textfraction{.2}
\def\floatpagefraction{.7}
\setcounter{dbltopnumber}{2}
\def\dbltopfraction{.8}
\def\dblfloatpagefraction{.8}
\def\dbltextfloatsep{0.8\textfloatsep}
\let\footnotesize=\small

\def\helvetica{\relax}
\def\timesroman{\relax}

\doublehyphendemerits5000
\hfuzz0pt
\tolerance=9999
\pretolerance=-1
\emergencystretch=25pt
\hbadness=30000
\hyphenpenalty=100
\bibleftmargin=1.2em
\def\bibnewpage{}
\def\bibsize{\small}
\def\bib@indent{\labelsep=0em
                \leftmargin=\bibleftmargin
                \itemindent=-\bibleftmargin}
                \def\newblock{\hskip .11em plus .33em minus .07em}\sloppy
}

%%%%%%%%%%%%%%%%%%
%                %
% REGULAR FORMAT %
%                %
%%%%%%%%%%%%%%%%%%

\@ifundefined{def@doc}{}{%
% Level One:   Centered Uppercase Heading
% Level Two:   Centered Heading
% Level Three: Centered Italics Heading
% Level Four:  Flushleft Italics Heading occuping a whole line
% Level Five:  Intended Italics Heading followed by texts in the same line
%
\def\levelone#1{\apa@startsection%
    {\z@}{\b@level@one@skip}{\e@level@one@skip}%
    {\leftskip4em plus 1fill\rightskip\leftskip\parindent\z@\relax%
     \uppercase\expandafter{\large #1}}}    % <- \bf
\def\leveltwo#1{\apa@startsection%
    {\z@}{\b@level@two@skip}{\e@level@two@skip}%
    {\leftskip4em plus 1fill\rightskip\leftskip\parindent\z@\relax%
     {\large #1}}}                           % <- \bf
\def\levelthree#1{\apa@startsection%
    {\z@}{\b@level@three@skip}{\e@level@three@skip}%
    {\leftskip4em plus 1fill\rightskip\leftskip\parindent\z@\relax%
    {\large\em #1}}}
\def\levelfour#1{\apa@startsection%
    {\z@}{\b@level@four@skip}{\e@level@four@skip}{{\em #1}}}
\def\levelfive#1{\apa@startsection%
    {\parindent}{\b@level@five@skip}{\e@level@five@skip}{{\em #1}.}\hskip .5em plus 1pt minus .5pt}

\long\def\title#1{\long\def\@title{#1}}
\long\def\author#1{\long\def\@author{#1}}
\long\def\twoauthors#1#2{\long\def\@authorOne{#1}\long\def\@authorTwo{#2}%
\long\def\@author{#1}}
\long\def\affiliation#1{\long\def\@affil{#1}}
\long\def\twoaffiliations#1#2{\long\def\@affilOne{#1}\long\def\@affilTwo{#2}%
\long\def\@affil{#1}}
\long\def\note#1{\long\def\@note{#1}}
\long\def\abstract#1{\long\def\@abstract{#1}}
\long\def\acknowledgements#1{\long\def\@acks{#1}}
\def\rightheader#1{\def\r@headr{\protect\uppercase{\protect\small #1}}}
\def\leftheader#1{\def\r@headl{\protect\uppercase{\protect\small #1}}}
\def\shorttitle#1{\def\s@title{#1}}
\def\rheadname{Running Head:}
\def\acksname{Author Note}
\pagestyle{myheadings}

\def\maketitle{
 \@ifundefined{@author}{\typeout{Warning: Author not defined} \def\@author{Author}}{}
 \@ifundefined{@title}{\typeout{Warning: Title not defined} \def\@title{Title}}{}
 \@ifundefined{@affil}{\typeout{Warning: Affiliation not defined} \def\@affil{Affiliation}}{}
 \@ifundefined{@abstract}{\typeout{Warning: Abstract not defined \def\abstract{}}}{}
 \@ifundefined{r@headr}{\typeout{Using title for running head} 
                        \def\r@headr{\protect\uppercase{\@title}}
                        \markright{\rm \@title \protect\\ \thepage}} {}
 \@ifundefined{r@headl}{\let\r@headl\r@headr}{}
 \@ifundefined{s@title}{\let\s@title\r@headr}{}
  \sloppy 
  \setlength{\parindent}{0.4in}
  \begin{center}
  \vspace*{0.3in}

  {\LARGE \@title}\\
  
  \vspace{0.3in}

  \@ifundefined{@authorTwo}{
  {\Large \@author} \\

  \@affil \vspace{0.1in} \\ }{
  {\Large \@authorOne} \\

  \@affilOne \vspace{0.1in} \\
  \@authorTwo \\

  \@affilTwo \vspace{0.1in} \\ }
  \@ifundefined{@note}
   {\vspace*{\baselineskip} }
   {\@note\vspace{0.2in}}

  {\bf Abstract}\vspace{0.1in}

  \parbox{5in}{\@abstract}\vspace{0.25in}
  \end{center}
 \pagenumbering{arabic}
 \thispagestyle{empty}
 \@ifundefined{@acks}
  {}
  {\begin{figure}[b]
   \parbox{\textwidth}{ \setlength{\parindent}{0.2in}
   \noindent \makebox[2in]{\hrulefill}

   \vspace*{0.05in}
   {\footnotesize
   \indent \@acks

   }}
   \end{figure}}
 \markboth{\hfill \r@headl \hfill}{\hfill \r@headr \hfill}
 \noindent
}

\ThreeLevelHeading
\def\refname{References}
\setlength{\topmargin}{-0.25in}
\setlength{\oddsidemargin}{0.25in}
\setlength{\evensidemargin}{0.25in}
\setlength{\textwidth}{6in}
\setlength{\textheight}{8.5in}
\setcounter{secnumdepth}{0}

\setcounter{topnumber}{2}
\def\topfraction{.7}
\setcounter{bottomnumber}{2}
\def\bottomfraction{.6}
\setcounter{totalnumber}{3}
\def\textfraction{.2}
\def\floatpagefraction{.7}
\setcounter{dbltopnumber}{2}
\def\dbltopfraction{.8}
\def\dblfloatpagefraction{.8}
\def\dbltextfloatsep{0.8\textfloatsep}

\def\helvetica{\relax}
\def\timesroman{\relax}
\def\centeredcaption#1{\caption}

\long\def\@makecaption#1#2{
 \vskip 10pt 
 \setbox\@tempboxa\hbox{#1 #2}
 \ifdim \wd\@tempboxa >\hsize #1 #2\par \else \hbox
to\hsize{\hfil\box\@tempboxa\hfil} 
 \fi 
 \vskip 4pt}

\long\def\@caption#1[#2]#3{\par\addcontentsline{\csname
  ext@#1\endcsname}{#1}{\protect\numberline{\csname 
  the#1\endcsname}{\ignorespaces #2}}\begingroup
    \@parboxrestore
    \normalsize
    \@makecaption{\csname fnum@#1\endcsname}{\ignorespaces #3}\par
  \endgroup}

\def\fnum@figure{\small {\em Figure {\thefigure}}.\hspace{0.07in}}
\long\def\fnum@table{\small Table {\thetable}:}

\bibleftmargin=2em
\def\bibnewpage{}
\def\bibsize{\small}
\def\bib@indent{\labelsep=0em
                \leftmargin=\bibleftmargin
                \itemindent=-\bibleftmargin}
                \def\newblock{\hskip .11em plus .33em minus .07em}\sloppy
}

%=======================================================

\long\def\ignore#1{\relax}
\let\oneLevelHeading=\OneLevelHeading
\let\twoLevelHeading=\TwoLevelHeading
\let\threeLevelHeading=\ThreeLevelHeading
\let\fourLevelHeading=\FourLevelHeading
\let\fiveLevelHeading=\FiveLevelHeading
\def\HeadingLevels#1{#1LevelHeading}
\def\headinglevels#1{#1LevelHeading}
