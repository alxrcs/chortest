--
-- Author: Emilio Tuosto <emilio.tuosto@gssi.it>
--

----
--
-- See GCGrammar.y for comments on the syntax
-- This parser compiles the .gc in latex using
-- the macros defined in ggmacros.tex
--
-- Adapt Misc.tabsize to the value of your editor 
--
-----

{
module GCtoLatex where
import Data.List as L (elem, length, filter, (\\))
import Misc
}

%name gc2latex
%tokentype { Token }
%monad { Ptype } { thenPtype } { returnPtype }
%lexer { lexer } { TokenEof }
%error { parseError }

%token
  str	        { TokenStr $$   }
  '(o)'         { TokenEmp }
  '->'	     	{ TokenArr     }
  '=>'	        { TokenMAr      }
  '='	     	{ TokenEq       }
  '|'	        { TokenPar      }
  '+'	        { TokenBra      }
  '%'	        { TokenGrd      }
  '*'	        { TokenSta      }
  ';'	        { TokenSeq      }
  '@'   	{ TokenUnt      }
  ':'	        { TokenSec      }
  ','	        { TokenCom      }
  '{'	        { TokenCurlyo   }
  '}'	        { TokenCurlyc   }
  '['	        { TokenCtxo     }
  ']'	        { TokenCtxc     }
  '&'	        { TokenAnd      }
  'sel'         { TokenSel 3    }
  'branch'      { TokenSel 6    }
  'repeat'      { TokenRep      }
  'unless'      { TokenUnl      }
  'let'         { TokenLet      }
  'in'          { TokenIn       }
  'do'          { TokenDo       }
  'with'        { TokenWith     }
  '[]'          { TokenHole     }

%right '|'
%right '+'
%right '%'
%right ';'
%right ','
%left '&'


%%


G :: { String }
G : GE
  {
    (fst $ $1 (1,1))
  }


GE :: { Position -> (String, Position) }
GE : E B
  {
    \p -> let
      (e, ee) = $1 p
      (b, eb) = $2 ee
      res = e ++ b
    in
       (res, eb)
  }
  | E B pos '|' GE
    {
      \p -> let
        (e, ee) = $1 p
        (b, eb) = $2 ee
        (g, eg) = $5 $3
        res = e ++ b ++
             (mkspace eb $3 1) ++ "\\gparop" ++ g
      in
        (res, eg)
    }


E :: { Position -> (String, Position) }
E : pos 'let' A pos 'in'
  {
    \p -> let
      (a,pa) = $3 $1
      res = (mkspace p $1 3) ++ (mklet a (mkspace pa $4 2))
    in
      (res, $4)
  }
  | {- empty -}
    { \p -> ("", p) }


A :: { Position -> (String,Position) }
A : pos str R pos '=' Ctx
    {
      \p -> let
        (r,pr) = $3 p
        (c,pc) = $6 $4
        res = (mkspace p $1 1) ++ (mksf $2 Const) ++
          r ++
          (mkspace pr $4 1) ++ (mksign "=") ++
          c
      in
        (res, pc)
    }
  | A pos '&' pos str R pos '=' Ctx
    {
      \p -> let
        (a,pa) = $1 p
        (r,pr) = $6 $4
        (c,pc) = $9 $7
        res = a ++
          (mkspace pa $2 1) ++ (mksign "&") ++
          (mkspace $2 $4 1) ++ (mksf $5 Const) ++
          r ++
          (mkspace pr $7 1) ++ (mksign "=") ++
          c
      in
        (res, pc)
    }


R :: { Position -> (String, Position) }
R : pos '@' fparams
  {
    \p -> let
      (r,er) = $3 $1
      res = (mkspace p $1 1) ++ (mksign "@") ++
        r
    in
      (res, er)
  }
  | {- empty -}
    { \p -> ("", p) }


Ctx :: { Position -> (String, Position) }
Ctx : pos '[]'
  {
    \p -> ((mkspace p $1 2) ++ "\\texttt{[]}", $1)
  }
  | pos 'do' call
    {
      \p -> let
          (c,pc) = $3 $1
          res = (mkspace p $1 0) ++ "\\gdokw" ++
            c
        in
          (res, pc)
    }
  | pos str pos '->' pos str pos ':' pos str
    {
      \p -> mkinteraction p $1 $2 $3 $5 $6 $7 $9 $10
    }
  | pos str pos '=>' pptps pos ':' pos str
  {
    \p -> let
      ps = $1
      s = $2
      pa = $3
      (r,pr) = $5 $3
      pdp = $6
      pm = $8
      m = $9
      res = (mkspace p ps 1) ++ (mksf s Ptp) ++
        (mkspace ps pa 2) ++ (mksign "=>") ++
        r ++
        (mkspace pr pdp 1) ++ (mksign ":") ++
        (mkspace pdp pm 1) ++ (mksf m Msg)
    in
      (res,pm)    
  }
  | choiceop pos '{' Brxs pos '}'
    {
      \p -> let
        (c, ec) = $1 p
        (b, pb) = $4 $2
        res = c ++ (mkspace ec $2 1) ++ (mksign "{") ++
          b ++
          (mkspace pb $5 1) ++ (mksign "}")
        in
        (res, $5)
    }
  | Ctx pos ';' Ctx
    {
      \p -> let
        (c1,p1) = $1 p
        (c2,p2) = $4 $2
        res = c1 ++
          (mkspace p1 $2 1) ++ "\\gseqop" ++ c2
        in
        (res, p2)
    }
  | Ctx pos '|' Ctx
    {
      \p -> let
        (c1,p1) = $1 p
        (c2,p2) = $4 $2
        res = c1 ++
          (mkspace p1 $2 1) ++ "\\gparop" ++
          c2
        in
        (res, p2)
    }
  | pos '*' Ctx pos '@' pos str
    {
      \p -> let
        pi = $1
        g = $3
        pa = $4
        ps = $6
        s = $7
        in
        mkiter p pi g pa ps s
    }
  | pos 'repeat' pos str pos '{' Ctx guard pos '}'
    {
      \p -> let
        pr = $1
        ps = $3
        s = $4
        po = $5
        c = $7
        g = $8
        pc = $9
        in
        mkrepeat p pr ps s po c g pc
    }
  | pos 'repeat' pos '{' Ctx guard pos '}'
    {
      \p -> let
        pr = $1
        ps = pzero
        s = ""
        po = $3
        c = $5
        g = $6
        pc = $7
        in
        mkrepeat p pr ps s po c g pc
    }
  | pos '{' Ctx pos '}'
    {
      \p -> mkblk p $1 $3 $4
    }


Brxs :: { Position -> (String, Position) }
Brxs : Ctx
  { $1 }
  | Ctx pos '+' Brxs
    {
      \p -> let
        (c,ec) = $1 p
        (b,eb) = $4 $2
        res = c ++
          (mkspace ec $2 0) ++ "\\gchoop" ++ b
      in
        (res, eb)
    }


B :: { Position -> (String, Position) }
B : S
  { $1 }
  | choiceop pos '{' Br pos '+' Bs pos '}'
    {
      \p -> let
        (o, eo) = $1 p
        (r, er) = $4 $2
        (s, es) = $7 $5
        res = o ++
          (mkspace eo $2 1) ++ (mksign "{") ++
          r ++ (mkspace er $5 1) ++ "\\gchoop" ++
          s ++ (mkspace es $8 1) ++ (mksign "}")
      in
        (res, $8)
    }


choiceop :: { Position -> (String, Position) }                                                                                                                    
choiceop : pos 'sel' pos str
  {
    \p -> let
      res = (mkspace p $1 3) ++ "\\gselkw" ++
        (mkspace $1 $3 1) ++ (mksf $4 Ptp)
    in
      (res,$3)
  }
  | pos 'branch' pos str
    {
      \p -> let
        res = (mkspace p $1 6) ++ "\\gbrakw" ++
          (mkspace $1 $3 1) ++ (mksf $4 Ptp)
      in
        (res, $3)
    }
  | pos 'sel'
    {
      \p -> ((mkspace p $1 3) ++ "\\gselkw", $1)
    }
  | pos 'branch'
    {
      \p -> ((mkspace p $1 6) ++ "\\gbrakw", $1)
    }


Bs :: { Position -> (String, Position) }
Bs : Br
  { $1 }
  | Br pos '+' Bs
    {
      \p -> let
        (r, er) = $1 p
        (s, es) = $4 $2
        res = r ++ (mkspace er $2 1) ++ "\\gchoop" ++ s
        in
        (res, es)
    }


Br :: { Position -> (String, Position) }
Br : S guard
  {
    \p -> let
      (s,es) = $1 p
      (g,eg) = $2 es
      in
      (s ++ g, eg)
  }


S :: { Position -> (String, Position) }
S : pos '(o)'
  {
    \p -> let
            res = (mkspace p $1 3) ++ "\\gempty"
          in
            (res, $1)
  }
  | B pos ';' B
    {
      \p -> let
        (b, e) = $1 p
        (b', e') = $4 $2
        res = b ++
          (mkspace e $2 1) ++ "\\gseqop" ++
          b'
      in
        (res, e')
    }
  | pos str pos '->' pos str pos ':' pos str
  {
    let
      ps = $1
      s = $2
      pa = $3
      pr = $5
      r = $6
      pdp = $7
      pm = $9
      m = $10
    in
      \p -> mkinteraction p ps s pa pr r pdp pm m
  }
  | pos str pos '=>' pptps pos ':' pos str
  {
    \p -> let
      ps = $1
      s = $2
      pa = $3
      (r,pr) = $5 $3
      pdp = $6
      pm = $8
      m = $9
      res = (mkspace p ps 1) ++ (mksf s Ptp) ++
        (mkspace ps pa 2) ++ (mksign "=>") ++
        r ++
        (mkspace pr pdp 1) ++ (mksign ":") ++
        (mkspace pdp pm 1) ++ (mksf m Msg)
    in
      (res,pm)
    
  }
  | pos 'do' call
    {
      \p -> let
          (c, ec) = $3 $1
          res = (mkspace p $1 2) ++ "\\gdokw" ++ c
        in
          (res, ec)
      }
  | pos '*' GE pos '@' pos str
  {
    \p -> let
        pr = $1
        g = $3
        pa = $4
        ps = $6
        s = $7
      in
        mkiter p pr g pa ps s
  }
  | pos 'repeat' pos '{' GE guard pos '}'
    {
      \p -> let
          pr = $1
          po = $3
          ge = $5
          g = $6
          pc = $7
        in
          mkrepeat p pr pzero "" po ge g pc
    }
  | pos 'repeat' pos str pos '{' GE guard pos '}'
    {
      \p -> let
          pr = $1
          ps = $3
          s = $4
          po = $5
          ge = $7
          g = $8
          pc = $9
        in
          mkrepeat p pr ps s po ge g pc
    }
  | pos '{' GE pos '}'
  {
    \p -> mkblk p $1 $3 $4
  }


call :: { Position -> (String, Position) }
call : pos str aparams
       {
         \p ->
           let
             (r,er) = $3 $1
             res = (mkspace p $1 1) ++ (mksf $2 Const) ++
               r
         in
           (res,er)
       }
     | pos str pos '[' GE pos ']' aparams
       {
         \p -> let
           (g, eg) = $5 $3
           (r, er) = $8 $6
           res = (mkspace p $1 1) ++
             (mkspace $1 $3 1) ++ (mksf $2 Const) ++
             (mkspace $1 $3 1) ++ (mksign "[") ++
             g ++
             (mkspace eg $6 1) ++ (mksign "]") ++
             r
         in
           (res, er)
       }


fparams :: { Position -> (String, Position) }
fparams : pos str
  {
    \p -> ((mkspace p $1 1) ++ (mksf $2 FParam), $1)
  }
  | pos str strs
    {
      \p -> let
              (s, es) = $3 $1 FParam
              res = (mkspace p $1 1) ++ (mksf $2 FParam) ++ s
            in
              (res, es)
    }


aparams :: { Position -> (String, Position) }
aparams : pos 'with' strs
  {
    \p -> let
            (s,ps) = $3 $1 AParam
            res = (mkspace p $1 4) ++ "\\gwithkw" ++ s
          in
            (res, ps)
  }
  | {- empty -}
    { \p -> ("", p) }


strs :: { Position -> IDSort -> (String, Position) }
strs : pos str
  {
    \p i -> ((mkspace p $1 1) ++ (mksf $2 i), $1)
  }
  | pos str pos strs
    {
      \p i -> let
              (s, es) = $4 $3 i
              res = (mkspace p $1 1) ++ (mksf $2 i) ++ s
            in
              (res, es)
    }


guard :: { Position -> (String, Position) }
guard : pos 'unless' pos str pos '%' pos str
  {
    \p -> let
      pu = $1
      ps = $3
      s = $4
      pp = $5
      ps' = $7
      s' = $8
    in
      mkunless p pu ps s pp ps' s'
  }
  | pos 'unless' pos str pos '%' pos str pos ',' guard
    {
      \p -> let
        pu = $1
        ps = $3
        s = $4
        pp = $5
        ps' = $7
        s' = $8
        (init, pi) = mkunless p pu ps s pp ps' s'
        (g,pg) = $11 $9
        res = init ++
          (mkspace pi ps' (length s')) ++ (mksf s' AParam) ++
          (mkspace ps' $9 1)
          ++ (mksign ",") ++ g
      in
        (res, pg)
    }
  | {- empty -}
    { \p -> ("", p) }


pptps :: { Position -> (String, Position) }
pptps : pos str
  {
    \p ->
      ((mkspace p $1 1) ++ (mksf $2 Ptp), $1)
  }
  | pos str pos pptps
  {
    \p -> let
      (r,er) = $4 $3
      res = (mkspace p $1 1) ++ (mksf $2 Ptp) ++
        (mkspace $1 $3 1) ++ (mksign ",") ++
        r
    in
      (res, er)
  }

pos :: { Position }
  : {- empty -}
  {% getPosition }

{

type Position = (Int, Int)

data Token =
  TokenStr String
  | TokenEmp
  | TokenArr
  | TokenPar
  | TokenBra
  | TokenSel Int
  | TokenGrd
  | TokenSeq
  | TokenRep
  | TokenSta
  | TokenUnt
  | TokenSec
  | TokenCom
  | TokenMAr
  | TokenUnl
  | TokenCurlyo
  | TokenCurlyc
  | TokenCtxo
  | TokenCtxc
  | TokenLet
  | TokenAnd
  | TokenIn
  | TokenDo
  | TokenWith
  | TokenEq
  | TokenHole
  | TokenEof
  deriving (Show)

data IDSort =
  Const
  | FParam
  | AParam
  | Ptp
  | Msg
  | Guard
  deriving (Eq)

lexer :: (Token -> Ptype a) -> Ptype a
lexer cont s p@(l, c) =
  case s of
    'b':'r':'a':'n':'c':'h':r ->
      cont (TokenSel 6) r (l,c+6)
    'r':'e':'p':'e':'a':'t':r ->
      cont TokenRep r (l,c+6)
    'u':'n':'l':'e':'s':'s':r ->
      cont TokenUnl r (l,c+6)
    'w':'i':'t':'h':r ->
      cont TokenWith r (l,c+4)
    '(':'o':')':r ->
      cont TokenEmp r (l, c+3)
    'l':'e':'t':r ->
      cont TokenLet r (l,c+3)
    's':'e':'l':r ->
      cont (TokenSel 3) r (l, c+3)
    'i':'n':r ->
      cont TokenIn r (l,c+2)
    'd':'o':r ->
      cont TokenDo r (l,c+2)
    '.':'.':r ->
      (lexer cont) (dropWhile (\c->c/='\n') r) (l+1, 1)
    '-':'>':r ->
      cont (TokenArr) r (l, c+2)
    '=':'>':r ->
      cont TokenMAr r (l, c+2)
    '[':']':r -> cont TokenHole r (l, c+2)
    '[':'[':r ->
      let
        takeComment acc s =
          case s of
            ']':']':_ -> acc
            _ -> takeComment (acc ++ [head s]) (tail s)
        tmp = takeComment "" r
        lskip = l + length (filter (\c -> c == '\n') tmp)
        cskip = 0 -- c + if lskip==0 then (length tmp) else 0
      in
        if tmp == r
        then
          Er ("Syntax error at <" ++
              (show $ l+1) ++ "," ++
              (show c) ++ ">: " ++
              "multiline comment not closed")
        else lexer cont (tail $ tail (r \\ tmp)) (lskip, cskip)
    x:r ->
      case x of
        '&' -> cont TokenAnd r (l, c+1)
        '*' -> cont TokenSta r (l, c+1)
        '%' -> cont TokenGrd r (l, c+1)
        '@' -> cont TokenUnt r (l, c+1)
        ':' -> cont TokenSec r (l, c+1)
        ';' -> cont TokenSeq r (l, c+1)
        '|' -> cont TokenPar r (l, c+1)
        '+' -> cont TokenBra r (l, c+1)
        ',' -> cont TokenCom r (l, c+1)
        '=' -> cont TokenEq r (l, c+1)
        '{' -> cont TokenCurlyo r (l, c+1)
        '}' -> cont TokenCurlyc r (l, c+1)
        '[' -> cont TokenCtxo r (l, c+1)
        ']' -> cont TokenCtxc r (l, c+1)
        ' ' -> (lexer cont) r (l, c+1)
        '\t' -> (lexer cont) r (l, c+Misc.tabsize)
        '\n' -> (lexer cont) r (l+1, 1)
        _ -> cont (TokenStr (fst s')) (snd s') (l, c + (length s'))
    [] ->
      (cont TokenEof) "" p
  where
    s' = span isAlpha s

data ParseResult a =
  Ok a
  | Er String
  deriving (Show)

type Ptype a =
  String -> Position -> ParseResult a

getPosition :: Ptype Position
getPosition _ p = Ok p

parseError token =
  \ _ p ->
    Er (synErr p token)
synErr :: Position -> Token -> String
synErr p@(l, c) token =
  "\\begin{verbatim} Syntax error at <" ++
  (show (l+1)) ++ "," ++ (show $ c+1) ++
  ">: " ++ err ++
  "\\end{verbatim}"
  where
    err =
      case token of
        TokenStr s  ->  "unexpected or malformed string: \'" ++ s ++
         "\'\n\t the following characters are forbidden:\
         \\'@\' \'.\' \',\' \';\' \':\' \'[\' \']\' \'{\' \'}\' \'|\' \'+\' \'*\' \'!\' \'?\' \'-\' \'%%\' \'§\'"
        TokenEmp    ->  "unexpected \'(o)\'"
        TokenArr    ->  "unexpected \'->\'"
        TokenPar    ->  "unexpected \'|\'"
        TokenBra    ->  "unexpected \'+\'"
        TokenSel o  ->  "unexpected " ++ (if o == 6 then "branch" else "sel")
        TokenGrd    ->  "unexpected \'unless\'"
        TokenSeq    ->  "unexpected \';\'"
        TokenRep    ->  "unexpected loop \'repeat\'"
        TokenSta    ->  "unexpected loop \'*\'"
        TokenUnt    ->  "unexpected \'@\'"
        TokenSec    ->  "unexpected \':\'"
        TokenCom    ->  "unexpected \',\'"
        TokenUnl    ->  "unexpected \'unless\' clause"
        TokenCurlyo ->  "unexpected \'{\'"
        TokenCurlyc ->  "unexpected \'}\'"
        TokenCtxo   ->  "unexpected \'[\'"
        TokenCtxc   ->  "unexpected \']\'"
        TokenLet    ->  "unexpected \'let\'"
        TokenAnd    ->  "unexpected \'&\'"
        TokenIn     ->  "unexpected \'in\'"
        TokenDo     ->  "unexpected \'do\'"
        TokenWith   ->  "unexpected \'with\'"
        TokenHole   ->  "unexpected \'[]\'"
        TokenEq     ->  "unexpected \'=\'"
        TokenEof    ->  "Perhaps an unexpected trailing symbol"

thenPtype :: Ptype a -> (a -> Ptype b) -> Ptype b
m `thenPtype` k = \s p ->
  case m s p of
    Ok v -> k v s p
    Er e -> Er e

returnPtype :: a -> Ptype a
returnPtype v = \_ _ -> Ok v

failPtype :: String -> Ptype a
failPtype err = \_ _ -> Er err

mktab c =
  if c < 2
  then ""
  else "\t" ++ (mktab (c - 2))

hspace :: Int -> String
hspace c =
    if c == 0
    then "\\ "
    else "\\hshift{" ++ (show c) ++ "}"

vspace :: Int -> String
vspace l =
  if l == 0 then "" else "\\vshift{" ++ (show l) ++ "}"

mkspace :: Position -> Position -> Int -> String
mkspace (l,c) (l',c') offset =
  let
    (h, tab) =
      if (l == l')
      then (if c' - offset - c < 2 then 0 else c' - offset - c + 1, "")
      else (c' - offset + 1, "\n" ++ (mktab $ c' - offset + 1))
    v = l' - l
  in
    (vspace v) ++ tab ++ (hspace h)
 
mksign :: String -> String
mksign s =
  case s of
    "" -> ""
    "->" -> "\\terminal{\\xrightarrow{}}"
    "{" -> "\\terminal{\\{}"
    "}" -> "\\terminal{\\}}"
    "%" -> "\\terminal{\\%}"
    "&" -> "\\terminal{\\&}"
    _ -> "\\separator{" ++ s ++ "}"

mkunless :: Position -> Position -> Position -> String -> Position -> Position -> String -> (String, Position)
mkunless p pu ps s pp ps' s' =
  let
    res = (mkspace p pu 6) ++ "\\gunlessop" ++
      (mkspace pu ps 1) ++ (mksf s Ptp) ++
      (mkspace ps pp 1) ++ (mksign "%") ++
      (mkspace pp ps' 1) ++ (mksf s' Guard)
  in (res, ps')

mkrec :: String -> String -> String
mkrec s atsign =
  "\\grecop" ++ (if s == "" then "" else atsign ++ (mksf s Ptp))

mksf :: String -> IDSort -> String
mksf s id
  | id == Const =
      "\\gconst{" ++ s ++ "}"
  | id == FParam =
      "\\gfparam{" ++ s ++ "}"
  | id == AParam =
      case s of
        "" -> ""
        _ ->
          if  L.elem (head s) ['A'..'Z']
          then mksf s Ptp
          else mksf s Msg
  | id == Ptp =
      "\\ptp[" ++ s ++ "]"
  | id == Msg =
      "\\msg[" ++ s ++ "]"
  | id == Guard =
      "\\aguard[" ++ s ++ "]"

mklet :: String -> String -> String
mklet s spc =
  if s == ""
  then ""
  else ("\\gletkw" ++ s ++ spc ++ "\\ginkw")

mkinteraction :: Position -> Position -> String -> Position -> Position -> String -> Position -> Position -> String -> (String, Position)
mkinteraction p ps s pa pr r pdp pm m =
 let
   res = (mkspace p ps 1) ++ (mksf s Ptp) ++
            (mkspace ps pa 2) ++ (mksign "->") ++
            (mkspace pa pr 1) ++ (mksf r Ptp) ++
            (mkspace pr pdp 1) ++ (mksign ":") ++
            (mkspace pdp pm 1) ++ (mksf m Msg)
  in
    (res,pm)

mkblk :: Position -> Position -> (Position -> (String, Position)) -> Position -> (String, Position)
mkblk p po body pc =
  let
    (c, ec) = body po
    res = (mkspace p po 1) ++ (mksign "{") ++
      c ++ (mkspace ec pc 1) ++ (mksign "}")
  in
    (res,pc)

mkiter :: Position -> Position -> (Position ->(String, Position)) -> Position-> Position -> String -> (String, Position)
mkiter p pi g pa ps s =
  let
    (c, ec) = g pi
    res = (mkspace p pi 1) ++ "\\grecop" ++
      c ++ (mkspace ec pa 1) ++ "\\grecopp" ++
      (mkspace pa ps 1) ++ (mksf s Ptp)
    in
      (res, ps)

mkrepeat :: Position -> Position -> Position -> String -> Position -> (Position -> (String, Position)) -> (Position -> (String, Position)) -> Position -> (String, Position)
mkrepeat p pr ps s po ge g pc =
  let
    (b,eb) = ge po
    (g', eg) = g eb
    res =
      case ps == pzero of
        True ->
          (mkspace p pr 6) ++ "\\grepkw" ++
          (mkspace pr po 1) ++ (mksign "{") ++
          b ++ g' ++
          (mkspace eb pc 1) ++ (mksign "}")
        False ->
          (mkspace p pr 6) ++ "\\grepkw" ++
          (mkspace pr ps 1) ++ (mksf s Ptp) ++
          (mkspace ps po 1) ++ (mksign "{") ++
          b ++ g' ++
          (mkspace eb pc 1) ++ (mksign "}")
  in
    (res,pc)
-- mkrepeat p pr pzero "" po c ge pc =
--   let
--     (b,eb) = c po
--     (g',eg) = ge eb
--     res = (mkspace p pr 6) ++ "\\grepkw" ++
--       (mkspace pr po 1) ++ (mksign "{") ++
--       b ++ g' ++ 
--       (mkspace eb pc 1) ++ (mksign "}")
--   in
--     (res, pc)
-- mkrepeat p pr ps s po ge g pc =
--   let
--     (b,eb) = ge po
--     (g', eg) = g eb
--     res = (mkspace p pr 6) ++ "\\grepkw" ++
--       (mkspace pr ps 1) ++ (mksf s Ptp) ++
--       (mkspace ps po 1) ++ (mksign "{") ++
--       b ++ g' ++
--       (mkspace eb pc 1) ++ (mksign "}")
--   in
--     (res,pc)

pzero = (0,0)

debugmsg :: [Position] -> String
debugmsg [] = ""
debugmsg (p:ps) = (show p) ++ (debugmsg ps)

}
