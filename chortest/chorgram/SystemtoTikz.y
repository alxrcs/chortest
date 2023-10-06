--
--
-- Author: Emilio Tuosto <emilio.tuosto@gssi.it>
--
----
--
-- Generating latex of CFSM; based on the SystemGrammar syntax 
--
----

{
module SystemtoTikz where
import Data.List as L
import qualified Data.Map.Strict as M
}

%name cfsmtotikz
%tokentype { Token }
%monad { Ptype } { thenPtype } { returnPtype }
%lexer { lexer } { TokenEof }
%error { parseError }

%token
  str	        { TokenStr $$ }
  lab	        { TokenLab $$ }
  ';'	        { TokenSep    }
  '!'	        { TokenSnd    }
  '?'	     	{ TokenRcv    }
  '('	        { TokenOrb    }
  ')'	        { TokenCrb    }
  "tau"         { TokenTau    }

%nonassoc '!' '?'
%right ';'


%%

M :: { (NodeMap -> String) -> String }
M : States ';' Transitions NodeLabels
  {
    \mknode -> let
      grid = $4 $1
      trxs = $3
    in
      "\n\\begin{tikzpicture}[mycfsm]\n" ++
        (mknode grid) ++
        "\n  %\n" ++ (mkedges trxs) ++
      "\n\\end{tikzpicture}"
  }
  
States :: { NodeMap }
States : {- empty -}
  {
    M.empty
  }
  | pos str States
  {
    let
      p = $1
      s = $2
    in
      M.insert s (hshift p (-(length s)), s, "") $3
  }

Transitions :: { [Edge] }
Transitions : {- empty -}
  {
    []
  }
  | str Label str Tikz Transitions
  {
    ($1, $2 $4, $3):$5
  }

Label :: { (String, String) -> String }
Label : '(' str str op str ')'
  {
  \t -> mklabel $2 $3 $4 $5 t
  }
  | '(' ')'
  {
    \t -> mklabel "" "" TokenEmp "" t
  }
  | '(' "tau" ')'
  {
    \t -> mklabel "" "" $2 "" t
  }


op : '!' { $1 }
  |  '?' { $1 }


NodeLabels :: { NodeMap -> NodeMap }
NodeLabels : {- empty -}
  {
    \m -> m
  }
  | ';' str Tikz NodeLabels
  {
    \m -> let
           (p,n,_) = m M.! $2
           (txt,tikz) = $3
          in
            $4 (M.adjust (\_ -> (p, txt, tikz)) $2 m)
  }

Tikz :: { (String, String) }
Tikz : {- empty -}
  { ("","") }
  | lab
  { $1 }

pos :: { Position }
  : {- empty-}
  {% getPosition }

{

-- Coordinates of a node in the grid
type Position = (Int, Int)

-- Transitions: (q,l,q') represent q -- l --> q'
type Edge = (String, String, String)

-- The following maps assigns information to node ids
-- id |-> (id_position, id_txt, id_tikz_modifiers)
type NodeMap = M.Map String (Position,String,String)

data Token =
  TokenStr String
  | TokenSep
  | TokenLab (String, String)
  | TokenEmp
  | TokenSnd
  | TokenRcv
  | TokenOrb
  | TokenCrb
  | TokenTau
  | TokenEof
  deriving Show

lexer :: (Token -> Ptype a) -> Ptype a
lexer cont s p@(l, c) =
    case s of
      't':'a':'u':r ->
        cont TokenTau r (hshift p (lenToken TokenTau))
      ';':r ->
        cont TokenSep r (hshift p (lenToken TokenSep))
      '%':r ->
        let
          aux c = L.elem c "%\n;"
          (id,s2) = break aux r
          (tikz,r') = break aux (tail s2)
        in
          cont (TokenLab (id,tikz)) r' (hshift p (length (id ++ tikz) +2))
      '!':r -> cont TokenSnd r (hshift p 1)
      '?':r -> cont TokenRcv r (hshift p 1)
      '(':r -> cont TokenOrb r (hshift p 1)
      ')':r -> cont TokenCrb r (hshift p 1)
      ' ':r -> (lexer cont) r (hshift p 1)
      '\t':r -> (lexer cont) r (hshift p 1)
      '\n':r -> (lexer cont) r (l+1, 1)
      [] -> (cont TokenEof) "" p
      _ ->
        let
          (s1,s2) = span isAlpha s
          isAlpha c =
            let
              allowed = ['0' .. 'z'] ++ "-$.,{}"
              forbidden = "#&~\"_@:;()[]|+*/^!?%§"
            in
              L.elem c ([x | x <- allowed, not (L.elem x forbidden)])
        in
          cont (TokenStr s1) s2 (l,c+(length s1))
          

data ParseResult a =
  Ok a
  | Er String
  deriving (Show)

type Ptype a =
  String -> Position -> ParseResult a

showToken :: Token -> String
showToken token =
  case token of
    TokenStr s -> s
    TokenLab (s,s') -> ('%':s ++ '%':s')
    TokenSep -> ";"
    TokenEmp -> ""
    TokenSnd -> "!"
    TokenRcv -> "?"
    TokenOrb -> "("
    TokenCrb -> ")"
    TokenTau -> "tau"
    TokenEof -> ""

lenToken t =
  case t of
    TokenStr s -> length s
    _ -> length (showToken t)

getPosition :: Ptype Position
getPosition _ p = Ok p

parseError token =
  \ _ p ->
    Er (synErr p token)

synErr :: Position -> Token -> String
synErr p token =
  "Syntax error at " ++ (show p) ++ ": " ++ err
  where
    err =
      case token of
        TokenStr s ->
          "unknown error in string '" ++ (showToken token) ++ "'"
        TokenLab s ->
          "unknown error in node reshaping '" ++ (showToken token) ++ "'"
        TokenEof -> "Perhaps an unexpected trailing symbol"
        _ ->
          "unexpected \\'" ++ (showToken token) ++ "\\'"

thenPtype :: Ptype a -> (a -> Ptype b) -> Ptype b
m `thenPtype` k = \s p ->
  case m s p of
    Ok v -> k v s p
    Er e -> Er e

returnPtype :: a -> Ptype a
returnPtype v = \_ _ -> Ok v

failPtype :: String -> Ptype a
failPtype err = \_ _ -> Er err

myErr :: String -> a
myErr err = error ("cfsm2tikz: ERROR - " ++ err)

{-
                  Auxiliary stuff for tikzing
-}

hshift :: Position -> Int -> Position
hshift (l,c) offset = (l,c+offset)

mklabel :: String -> String -> Token -> String -> (String, String) -> String
mklabel p q t m (t1,t2) =
  let
    action =
      case t of
        TokenSnd -> "\\aout[" ++ p ++ "][" ++ q ++"][][" ++ m ++ "]"
        TokenRcv -> "\\ain[" ++ p ++ "][" ++ q ++"][][" ++ m ++ "]"
        TokenTau -> "\\tau"
        TokenEmp -> ""
    tikzn =
      if t2 == ""
      then "above"
      else t2
    tikze =
      if t1 == ""
      then ""
      else "[" ++ t1 ++ "]"
  in
    "edge" ++ tikze ++" node[" ++ tikzn ++ "]{\\ensuremath{" ++ action ++ "}}"

tikzPosition :: Position -> String
tikzPosition (l,c) =
  show (c,1-l)

nodePos :: String -> NodeMap -> Position
nodePos n m =
  p where (p,_,_) = m M.! n
  
nodeText :: String -> NodeMap -> String
nodeText n m =
  let
    (_,s,_) = m M.! n
  in
  " {" ++
  (case s of
    "" -> ""
    c:[]-> "\\ensuremath{" ++ s ++ "}"
    _ -> "\\ensuremath{\\mathit{" ++ s ++ "}}") ++
  "};\n"

nodeTikz :: String -> NodeMap -> String
nodeTikz n m =
  if s == ""
  then ""
  else ", " ++ s
    where (_,_,s) = m M.! n
  
mkabsolute :: NodeMap -> String
mkabsolute m =
  let
    nodes = M.keys m
    aux n =
      let
        (p, nid, _) = m M.! n
        txt = 
          case (length nid) of
            0 -> ""
            1 -> "\\ensuremath{" ++ nid ++ "}"
            _ -> "\\ensuremath{\\mathit{" ++ nid ++ "}}"
      in
        "  \\node[state" ++ (nodeTikz n m) ++ "] (" ++ n ++ ") at " ++
        (tikzPosition p) ++ " {" ++ txt ++ "};"
  in
    L.intercalate "\n" $ L.map aux nodes

mkrelative :: NodeMap -> String
mkrelative m =
  let
    tmp = L.sort $ L.map (\n -> (nodePos n m, n)) (M.keys m)
    (p0,q0) = head tmp
    m' = M.fromList $ tmp
    start =
      "  \\node[state" ++ (nodeTikz q0 m) ++
      "] (" ++ q0 ++ ") at " ++ (show p0) ++ 
      (nodeText q0 m)
    aux p@(pl, _) p' ns =
      case ns of
        [] -> ""
        n:ns' ->
          let
            ((l,c),_,_) = m M.! n
            mod = "state," ++
              if l == pl
              then "right = of " ++ (m' M.! p') ++ (nodeTikz n m)
              else "below = of " ++ (m' M.! p) ++ (nodeTikz n m)
          in
            "  \\node[" ++ mod ++ "] (" ++ n ++ ") " ++
            (nodeText n m) ++
            (if l == pl then aux p (l,c) ns' else aux (l,c) (l,c) ns')
  in start ++ (aux p0 p0 (tail $ M.elems m'))

mkedges :: [SystemtoTikz.Edge] -> String
mkedges es =
  let
    aux (s,l,t) =
      "  \\path (" ++ s ++ ") " ++
      l ++ " (" ++ t ++ ");"
  in
    L.intercalate "\n" $ L.map aux es

verbose file = "-- [Ignoring: " ++ file ++ " because of the -v option.]\n--\n-- " ++
  L.foldr (\x s -> x ++ "\n-- " ++ s) "" [
	"-- csfm2tikz generates tikz code from a file formatted as some ';'-sections. Here is an example\n--",
	"   a    b",
	"      foo",
	"             d",
	"\n--   ;\n-- ",
	"   a (A  b ! int) b",
	"   a (tau) c   %red, bend left =20%below",
	"   c () d",
	"   d (b c ? str) b",
	"\n--   ;\n-- ",
	"   a %start%initial, initial where = left;",
	"   d %%;",
	"   b % b % yshift=3em;",
	"   c % c\n--",
	"The first section ends before the first ';' symbol, after which the second section starts and likewise for the second and third section.",
	"Section 1 lists node ids (\"a\", \"b\", \"foo\", \"d\"). Spacing is important as it determines the positioning of the nodes in the tikz picture.",
	"Section 2 lists transitions (in the example an output from \"a\" to \"b\", a tau-transition from \"a\" to \"c\", one with no labels from \"c\" to \"d\", and finally an input transition from \"b\" to \"d\". The tau-transition modifies the tikz edge and label node as specified by the '%'-separated text decorating the tau-transition.",
	"Section 3 is optional and it is made of ';' separated lines formatted as  \"% text % ... ;\". Each line specifies the text inside a node (e.g., the text of \"a\" is \"start\") and tikz modifiers. In the example \"a\" is the initial node, the text of \"d\" is empty and \"b\" shifted vertically.",
	"Tikz modifiers are optional as for node \"c\" above. The file must end with no spaces or empty lines."
  ]

}


