--
-- Author: Emilio Tuosto <emilio.tuosto@gssi.it>
--

-- COMMENT MOVED AT THE END OF THE FILE


{
module GCParser where
import SyntacticGlobalChoreographies
import ErlanGC
import Data.Set as S (empty, insert, union, fromList, toList, member, Set)
import qualified Data.Set as S (map)
import Data.List as L
import qualified Data.Map as M
import Misc
import CFSM
}

%name gcgrammar
%tokentype { Token }
%monad { Ptype } { thenPtype } { returnPtype }
%lexer { lexer } { TokenEof }
%error { parseError }

%token
  qstr	        { TokenQStr $$  }
  str	        { TokenStr $$   }
  '->'	     	{ TokenArr      }
  '=>'	        { TokenMAr      }
  '='	     	{ TokenEq       }
  '|'	        { TokenPar      }
  '+'	        { TokenBra      }
  '%'	        { TokenGrd      }
  '*'	        { TokenSta      }
  ';'	        { TokenSeq      }
  '@'   	{ TokenUnt      }
  ':'	        { TokenSec      }
  '{'	        { TokenCurlyo   }
  '}'	        { TokenCurlyc   }
  '['	        { TokenCtxo     }
  ']'	        { TokenCtxc     }
  '&'	        { TokenAnd      }
  '(o)'         { TokenEmp      }
  '[]'          { TokenHole     }
  'sel'         { TokenSel 3    }
  'branch'      { TokenSel 6    }
  'repeat'      { TokenRep      }
  'unless'      { TokenUnl      }
  'let'         { TokenLet      }
  'in'          { TokenIn       }
  'do'          { TokenDo       }
  'with'        { TokenWith     }

%right '|'
%right '+'
%right '%'
%right ';'
%left '&'


%%


G :: { (GC, Set Ptp) }
G : GE  { $1 emptyEnv }
  
GE :: { GCEnv -> (GC, Set Ptp) }
GE : E B
     { \env ->
         let
           join = compEnv env $1
         in
           $2 join
     }
   | E B '|' GE
     { \env ->
         let
           join = compEnv env $1
           (g, ptps) = $2 join
         in
           (Par (checkToken TokenPar (Par [g, fst ($4 join)])),
             S.union ptps (snd ($4 join))
           )
     }


E :: { GCEnv }
E : 'let' A 'in' { $2 emptyEnv }
  | {- empty -}  { emptyEnv }


A :: { GCEnv -> GCEnv }
A : str R '=' Ctx
    { \env -> (insEnv $1 $2 $4 env) }
  | A '&' str R '=' Ctx
    { \env -> insEnv $3 $4 $6 (compEnv env ($1 emptyEnv)) }


R :: { [String] }
R : '@' fparams { $2 }
  |  {- empty -}  { [] }


Ctx :: { GCEnv -> GCCtx }
Ctx : '[]'  { \_ -> Hole }
    | 'do' call
      { \env ->
          let
            (const, (g, _), params) = $2 env
            (ctx, formals) = env M.! const 
            m = M.fromList (L.zip formals params)
            g' = applyEnvGC env g
          in
            case (checkParameters formals params, M.member const env) of
              (Just err, _) -> myErr err
              (_, True) -> substx m $ fillx env g' ctx
              _ -> myErr ("Use of undefined constant: " ++ const)
       }
    | str '->' str ':' str qos
      { \_ ->
          case ((isPtp $1), (isPtp $3), ($1 == $3)) of
            (True, True, False) ->
              Actx ($1 , $3) ($5 ++ $6)
            (False, False, _) ->
              myErr ("Malformed participant names " ++ $1 ++ " and " ++ $3)
            (False, True, True) ->
              myErr ("Malformed participant name " ++ $1 ++ " and sender and receiver must be different")
            (False, True, False) ->
              myErr ("Malformed participant name " ++ $1)
            (True, False, False) ->
              myErr ("Malformed participant name " ++ $3)
            (_, _, True)         ->
              myErr ("Participant " ++ $1 ++ " cannot be both sender and receiver of a same interaction")
      }
    | str '=>' pptps ':' str qos
      { \_ ->
          case ($1 € $3, isPtp $1, $3) of
            (True, _, _) ->
              myErr ($1 ++ " must NOT be one of the receivers " ++ (mkSep $3 ", "))
            (False, False, _) ->
              myErr ("Malformed participant name " ++ $1)
            (False, True, []) ->
              myErr ("No receiver for " ++ $1)
            (False, True, s:[]) ->
              Actx ($1 , s) ($5 ++ $6)
            _ ->
              Parx (L.map (\s -> (Actx ($1 , s) $5)) $3)
      }
    | choiceop '{' Brxs '}'
      { \env ->
          let
            range = [1 .. length $3]
            brcs = M.fromList $ L.zip range (L.map (\x -> x env) $3)
          in
            Brax $1 (M.fromList $ L.zip range (checkTokenx TokenBra (Brax $1 brcs)))
      }
    | Ctx ';' Ctx
      { \env ->
          Seqx (checkTokenx TokenSeq (Seqx [$1 env, $3 env]))
      }
    | Ctx '|' Ctx
      { \env ->
          Parx (checkTokenx TokenPar (Parx [$1 env, $3 env]))
      }
    | '*' Ctx '@' str
      { \env ->
          if (isPtp $4)
          then Repx $4 ($2 env)
          else myErr ("Malformed participant name: " ++ $4)
      }
    | 'repeat' str '{' Ctx guard '}'
      { \env ->
          if (isPtp $2)
          then (Repx $2 ($4 env))
          else myErr ("Malformed participant name: " ++ $2)
      }
    | 'repeat' '{' Ctx guard '}'
      {
        \env -> Repx "" ($3 env)
      }
    | '{' Ctx '}'  { $2 }


Brxs :: { [GCEnv -> GCCtx] }
Brxs : Ctx  { [$1] }
     | Ctx '+' Brxs  { $1:$3 }


B :: { GCEnv -> (GC, Set Ptp) }
B : S  { $1 }
  | choiceop '{' Br '+' Bs '}'
    { \env ->
        (let
            branches = L.map fst ([$3 env] ++ ($5 env))
            aux g l = (checkToken TokenBra (fst g)) ++ l
            tmp = L.foldr aux [] branches
            gcs = M.fromList $ L.zip [1 .. length tmp] tmp
          in
           Bra $1 gcs,
          ptpsBranches ([$3 env] ++ ($5 env))
        )
    }


qos :: { String }
qos : {- -}
  {
    ""
  }
  | '{' qstr '}'
  {
    "{ " ++ $2 ++ " }"
  }  


{- TODO: selector should not be ignore -}
choiceop :: { Ptp }
choiceop : 'sel' str {$2}
         | 'branch' str {$2}
         | 'sel' {""}
         | 'branch' {""}


Bs :: { GCEnv -> [((GC, Set Ptp), M.Map String String  {- TODO: implement guards -} )] }
Bs : Br         { \env -> [$1 env] }
   | Br '+' Bs  { \env -> [$1 env] ++ ($3 env) }


Br :: { GCEnv -> ((GC, Set Ptp), M.Map String String) }
Br : S guard { \env -> checkGuard ($1 env) $2 }


S :: { GCEnv -> (GC, Set Ptp) }
S : '(o)'  { \_ -> (Emp, S.empty) }
  | B ';' B
    { \env ->
        let
          (b1, ptps1) = ($1 env)
          (b2, ptps2) = ($3 env)
        in
          (Seq (checkToken TokenSeq (Seq [b1, b2])),
           S.union ptps1 ptps2
          )
    }
  | str '->' str ':' str qos
  { \_ ->
      case ((isPtp $1), (isPtp $3), ($1 == $3)) of
        (True, True, False)  -> ((Act ($1 , $3) ($5 ++ $6)), S.fromList [$1,$3])
        (False, False, _) ->
          myErr ("Malformed participant names: " ++ $1 ++ " and " ++ $3)
        (False, True, True) ->
          myErr ("Malformed participant name: " ++ $1 ++ " and sender and receiver must be different")
        (False, True, False) ->
          myErr ("Malformed participant name: " ++ $1)
        (True, False, False) ->
          myErr ("Malformed participant name: " ++ $3)
        (_, _, True) ->
          myErr ("Participant " ++ $1 ++ " cannot be both sender and receiver of a same interaction")
  }
  | str '=>' pptps ':' str qos
    { \_ ->
        case ($1 € $3, isPtp $1, $3) of
          (True, _, _) ->
            myErr ($1 ++ " must NOT be one of the receivers " ++ (mkSep $3 ", "))
          (False, False, _) ->
            myErr ("Malformed participant name: " ++ $1)
          (False, True, []) ->
            myErr ($1 ++ " cannot be empty")
          (False, True, s:[]) ->
            ((Act ($1 , s) ($5 ++ $6)), S.fromList([$1,s]))
          _ ->
            (Par (L.map (\s -> (Act ($1 , s) $5)) $3), S.fromList($1:$3))
    }
  | 'do' call
    { \env ->
        let
          (const, (g, ptps), params) = $2 env
          (ctx, formals) = (env M.! const)
          m = M.fromList (L.zip formals params)
          g' = subst m (fill env (applyEnvGC env g) ctx)
          ptps' = S.map (rename m) $ S.union ptps (ctxptps ctx)
        in
          case (checkParameters formals params, M.member const env) of
            (Just err, _) -> myErr err
            (_, True) -> (g', ptps')
            _ -> myErr ("Use of undefined constant: " ++ const)
    }
  | '*' GE '@' str
    { -- Note the difference with Ctx on the checks
      \env ->
        let
          (g, ptps) = $2 env
        in
          case ((isPtp $4), ( $4 == "" || S.member $4 ptps)) of
            (True, True) ->
              (Rep $4 g , S.insert $4 ptps)
            (False, _) ->
              myErr ("Malformed participant name: " ++ $4)
            (True, False) ->
              myErr $ "Participant " ++ $4 ++
              " is not among the loop's participants: " ++
              (show $ toList ptps)
    }
  | 'repeat' str '{' GE guard '}'
    { \env ->
        let
          (g, ptps) = $4 env
        in
          case ((isPtp $2), (S.member $2 ptps)) of
            (True, True) ->
              (Rep $2 g , S.insert $2 ptps)
            (False, _) ->
              myErr ("Malformed participant name: " ++ $2)
            (True, False) ->
              myErr $ "Participant " ++ $2 ++
              " is not among the loop's participants: " ++
              (show $ toList ptps)
    }
  | 'repeat' '{' GE guard '}'
    { \env ->
        let
          (g, ptps) = $3 env
        in
          (Rep "" g , ptps)
    }
  | '{' GE '}'  { $2 }


call :: { GCEnv -> (GCConst, (GC, Set Ptp), [String]) }
call : str aparams
       { \_ -> ($1, (Emp, S.empty), $2) }
     | str '[' GE ']' aparams
       { \env -> ($1, $3 env, $5) }


fparams :: { [String] }
fparams : str
  {
    [$1]
  }
  | str strs
    {
      $1:$2
    }


aparams :: { [String] }
aparams : 'with' strs  { $2 }
       | {- empty -}  { [] }


strs :: { [String] }
strs : str  { [$1] }
     | str strs  { $1:$2 }

guard :: { M.Map String String }
guard : 'unless' str '%' str
        { M.insert $2 $4 M.empty }
      | 'unless' str '%' str guard
        { M.insert $2 $4 $5 }
      | {- empty -} {M.empty}


pptps :: { [String] }
pptps : str
       { if (isPtp $1)
         then [$1]
         else myErr ("Malformed participant name: " ++ $1)
       }
     | str pptps
       { if (isPtp $1)
         then
           case $2 of
             [] ->  [$1]
             s:l -> ($1:s:l)
         else myErr ("Malformed participant name: " ++ $1)
       }

{
data Token =
  TokenStr String
  | TokenQStr String
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

lexer :: (Token -> Ptype a) -> Ptype a
lexer cont s (l, c) (l',c') =
  -- (l,c) is the currently reached position in the parsing
  -- (l',c') is the position of the last accepted token
  case s of
    'b':'r':'a':'n':'c':'h':x:r ->
      case x of
        ' '  -> cont (TokenSel 6) r (l, (c+7)) (l, c)
        '\t' -> cont (TokenSel 6) r (l, (c+7)) (l, c)
        '{'  -> cont (TokenSel 6) ('{':r) (l, (c+6)) (l, c)
        '\n' -> cont (TokenSel 6) r ((l+1), 0) (l, c)
        _ -> (cont (TokenStr s')) r' (l, c + (length s')) (l, c)
    'r':'e':'p':'e':'a':'t':x:r ->
      case x of
        ' '  -> cont TokenRep r (l, (c+7)) (l, c)
        '\t' -> cont TokenRep r (l, (c+7)) (l, c)
        '{'  -> cont TokenRep ('{':r) (l, (c+6)) (l, c)
        '\n' -> cont TokenRep r ((l+1), 0) (l, c)
        _ -> (cont (TokenStr s')) r' (l, c + (length s')) (l, c)
    'u':'n':'l':'e':'s':'s':x:r ->
      case x of
        ' '  -> cont TokenUnl r (l, (c+7)) (l, c)
        '\t' -> cont TokenUnl r (l, (c+7)) (l, c)
        '{'  -> cont TokenUnl ('{':r) (l, (c+6)) (l, c)
        '\n' -> cont TokenUnl r ((l+1), 0) (l, c)
        _ -> (cont (TokenStr s')) r' (l, c + (length s')) (l, c)
    'l':'e':'t':x:r ->
      case x of
        ' '  -> cont TokenLet r (l, (c+4)) (l, c)
        '\t' -> cont TokenLet r (l, (c+4)) (l, c)
        '{'  -> cont TokenLet ('{':r) (l, (c+3)) (l, c)
        '\n' -> cont TokenLet r ((l+1), 0) (l, c)
        _ -> (cont (TokenStr s')) r' (l, c + (length s')) (l, c)
    's':'e':'l':x:r ->
      case x of
        ' '  -> cont (TokenSel 3) r (l, (c+4)) (l, c)
        '\t' -> cont (TokenSel 3) r (l, (c+4)) (l, c)
        '{'  -> cont (TokenSel 3) ('{':r) (l, (c+3)) (l, c)
        '\n' -> cont (TokenSel 3) r ((l+1), 0) (l, c)
        _ -> (cont (TokenStr s')) r' (l, c + (length s')) (l, c)
    'i':'n':x:r ->
      case x of
        ' '  -> cont TokenIn r (l, (c+3)) (l, c)
        '\t' -> cont TokenIn r (l, (c+3)) (l, c)
        '{'  -> cont TokenIn ('{':r) (l, (c+2)) (l, c)
        '\n' -> cont TokenIn r ((l+1), 0) (l, c)
        _ -> (cont (TokenStr s')) r' (l, c + (length s')) (l, c)
    'd':'o':x:r ->
      case x of
        ' '  -> cont TokenDo r (l, (c+3)) (l, c)
        '\t' -> cont TokenDo r (l, (c+3)) (l, c)
        '{'  -> cont TokenDo ('{':r) (l, (c+2)) (l, c)
        '\n' -> cont TokenDo r ((l+1), 0) (l, c)
        _ -> (cont (TokenStr s')) r' (l, c + (length s')) (l, c)
    'w':'i':'t':'h':x:r ->
      case x of
        ' '  -> cont TokenWith r (l, (c+5)) (l, c)
        '\t' -> cont TokenWith r (l, (c+5)) (l, c)
        '{'  -> cont TokenWith ('{':r) (l, (c+4)) (l, c)
        '\n' -> cont TokenWith r ((l+1), 0) (l, c)
        _ -> (cont (TokenStr s')) r' (l, c + (length s')) (l, c)
    's':'q':'o':'s':r ->
      let
        (q,r') = L.span (\x -> x /= '}') r
      in
        (cont (TokenQStr $ "sqos" ++ q)) r' (l, c + (length q)) (l, c)
    'r':'q':'o':'s':r ->
      let
        (q,r') = L.span (\x -> x /= '}') r
      in
        (cont (TokenQStr $ "rqos" ++  q)) r' (l, c + (length q)) (l, c)
    '(':'o':')':r ->
      cont TokenEmp r (l, (c+3)) (l, c)
    '.':'.':r ->
      (lexer cont) (dropWhile (\c->c/='\n') r) (l, 0) (l',c')
    '-':'>':r ->
      cont TokenArr r (l, (c+2)) (l, c)
    '=':'>':r ->
      cont TokenMAr r (l, (c+2)) (l, c)
    '[':']':r -> cont TokenHole r (l, (c+2)) (l, c)
    '[':'[':r ->
      let
        takeComment acc s =
          case s of
            ']':']':_ -> acc
            _ -> takeComment (acc ++ [head s]) (tail s)
        tmp = takeComment "" r
        lskip = l + L.length (L.filter (\c -> c == '\n') tmp)
        cskip = 0 -- c + if lskip==0 then (length tmp) else 0
      in
        if tmp == r
        then Er ("Syntax error at <" ++ (show $ l+1) ++ "," ++ (show $ c) ++ ">: " ++ "multiline comment not closed")
        else lexer cont (tail $ tail (r \\ tmp)) (lskip, cskip) (l', c')
    x:r ->
      case x of
        '&' -> cont TokenAnd r (l, (c+1)) (l, c)
        '*' -> cont TokenSta r (l, (c+1)) (l, c)
        '%' -> cont TokenGrd r (l, (c+1)) (l, c)
        '@' -> cont TokenUnt r (l, (c+1)) (l, c)
        ':' -> cont TokenSec r (l, (c+1)) (l, c)
        ';' -> cont TokenSeq r (l, (c+1)) (l, c)
        '|' -> cont TokenPar r (l, (c+1)) (l, c)
        '+' -> cont TokenBra r (l, (c+1)) (l, c)
        '=' -> cont TokenEq r (l, (c+1)) (l, c)
        '{' -> cont TokenCurlyo r (l, (c+1)) (l, c)
        '}' -> cont TokenCurlyc r (l, (c+1)) (l, c)
        '[' -> cont TokenCtxo r (l, (c+1)) (l, c)
        ']' -> cont TokenCtxc r (l, (c+1)) (l, c)
        ' ' -> (lexer cont) r (l, (c+1)) (l', c')
        '\t' -> (lexer cont) r (l, (c+1)) (l', c')
        '\n' -> (lexer cont) r ((l+1), 0) (l', c')
        _ -> (cont (TokenStr s')) r' (l, c + (length s')) (l, c)
    [] ->
      (cont TokenEof) "" (l, c) (l',c')
  where
    (s',r') = span (\c -> not (c € separators)) s
  

data ParseResult a =
  Ok a
  | Er String
  deriving (Show)

type Ptype a = String -> (Int, Int) -> (Int, Int) -> ParseResult a

parseError token =
  \ _ _ (l, c) ->
    Er (synErr l c token)

thenPtype :: Ptype a -> (a -> Ptype b) -> Ptype b
m `thenPtype` k = \s (l, c) (l', c') ->
  case m s (l, c) (l', c') of
    Ok v -> k v s (l, c) (l',c')
    Er e -> Er e

returnPtype :: a -> Ptype a
returnPtype a = \s _ _ -> Ok a

failPtype :: String -> Ptype a
failPtype err = \_ _ _ -> Er err


-- GC specific stuff

-- G-choreographies constants and contexts

type GCConst = String


type GCEnv = M.Map GCConst (GCCtx, [String])
emptyEnv :: GCEnv 
emptyEnv = M.empty

applyEnvGC :: GCEnv -> GC -> GC
applyEnvGC env g =
  {- PRE: for all x in M.keys env, env ! x is a constant-free context

     POST: substitute the uses in g of constants in M.keys env with
           the corresponding definition in env and passes parameters
  -}
  (L.foldr (\const -> aux const) g (M.keys env))
    where
      aux :: GCConst -> GC -> GC
      aux const = \c ->
        case c of
          Par gs -> Par (checkToken TokenPar (Par (L.map (applyEnvGC env) gs)))
          Bra p brc ->
            let
              tmp = L.foldr (++) [] (L.map (checkToken TokenBra) (M.elems brc))
              gcs = M.fromList $ L.zip [1 .. length tmp] tmp
            in
              Bra p gcs
          Seq gs -> Seq (checkToken TokenSeq (Seq (L.map (applyEnvGC env) gs)))
          Rep ptp g' -> Rep ptp (applyEnvGC env g')
          _ -> g


data GCCtx = Hole
  | Dox GCConst ([String]) GCCtx
  | Parx [GCCtx]
  | Brax Ptp (M.Map Label GCCtx)
  | Seqx [GCCtx]
  | Repx Ptp GCCtx
  | Actx Channel Message
  | Empx
           deriving (Show)


compEnv :: GCEnv -> GCEnv -> GCEnv
compEnv env env' =
  let
    common = M.intersection env env'
  in
    if (M.null common)
    then M.union env env'
    else myErr ("Double definition of constants: " ++ (mkSep (M.keys common) ", "))


compCtx :: (GCCtx, [String]) -> ([String]) -> GCCtx -> GCCtx
compCtx (ctx, formals) params ctx' =
-- replace the holes in ctx with ctx' and formal parameters with params
  case checkParameters formals params of
    Just err -> myErr err
    _ ->
      let
         m = M.fromList (L.zip formals params)
       in
         case ctx of
           Hole -> ctx'
           Parx ctxs -> Parx (L.map (\c -> compCtx (c, formals) params ctx') ctxs)
           Brax p ctxs -> Brax p (M.map (\c -> compCtx (c, formals) params ctx') ctxs)
           Seqx ctxs -> Seqx (L.map (\c -> compCtx (c, formals) params ctx') ctxs)
           Repx ptp ctx'' -> Repx ptp (compCtx (ctx'', formals) params ctx')
           Actx (s,r) msg -> Actx (rename m s, rename m r) (rename m msg)
           Dox const params' ctx'' -> Dox const (L.map (rename m) params') ctx''

checkParameters :: [String] -> [String] -> Maybe String
checkParameters formals params =
  case (formals, params) of
    (p:formals', p':params') ->
      if (isPtp p) && (isPtp p')
      then checkParameters formals' params'
      else Just ("Participants cannot be messages and viceversa: " ++ (show p) ++ (show p'))
    ([], _:_) -> Just ("Not enough formal parameters: " ++ (mkSep params ", ") ++ " cannot replace " ++ (mkSep formals ", "))
    (_:_, []) -> Just ("Not enough actual parameters: " ++ (mkSep params ", ") ++ " cannot replace " ++ (mkSep formals ", "))
    _ -> Nothing

applyEnvCtx :: GCEnv -> GCCtx -> GCCtx
applyEnvCtx env ctx =
  {- PRE: for all x in M.keys env, env ! x is a constant-free context

     POST: substitute the uses in ctx of constants in M.keys env with
           the corresponding definition in env
  -}
  L.foldr (\const -> aux const) ctx (M.keys env)
    where
      aux :: GCConst -> GCCtx -> GCCtx
      aux const = \c ->
        case c of
          Dox const params ctx' ->
            aux const (compCtx (env M.! const) params ctx')
          Dox _ _ _ -> myErr ("Use of undefined constant: " ++ const)
          Parx ctxs -> Parx (L.map (applyEnvCtx env) ctxs)
          Brax p brc -> Brax p (M.map (applyEnvCtx env) brc)
          Seqx ctxs -> Seqx (L.map (applyEnvCtx env) ctxs)
          Repx ptp ctx' -> Repx ptp (applyEnvCtx env ctx')
          Actx _ _ -> c
          Empx -> c
          _ -> c

substx :: M.Map String String -> GCCtx -> GCCtx
substx m ctx =
  case ctx of
    Dox const params ctx' -> Dox const (L.map (rename m) params) (substx m ctx')
    Parx ctxs -> Parx (L.map (substx m) ctxs)
    Brax p brc -> Brax p (M.map (substx m) brc)
    Seqx ctxs -> Seqx (L.map (substx m) ctxs)
    Repx ptp ctx' -> Repx ptp (substx m ctx')
    Actx (s,r) msg -> Actx (rename m s, rename m r) (rename m msg)
    Empx -> Empx

subst :: M.Map String String -> GC -> GC
subst m g =
  case g of
    Par gcs -> Par (L.map (subst m) gcs)
    Bra p brc -> Bra p (M.map (subst m) brc)
    Seq gcs -> Seq (L.map (subst m) gcs)
    Rep ptp gc' -> Rep ptp (subst m gc')
    Act (s,r) msg -> Act (rename m s, rename m r) (rename m msg)
    Emp -> Emp

insEnv :: GCConst -> ([String]) -> (GCEnv -> GCCtx) -> GCEnv -> GCEnv
insEnv const params absCtx env =
  {- PRE: for all x in M.keys env, env ! x is a constant-free context

     POST: insert in env the constant-free version of (absCtx env)
           obtained by replacing each x in M.keys env with its
           corresponding definition in env;
  -}
  if (M.member const env)
  then myErr ("Double definition of constant " ++ const)
  else
    let
      ctx = absCtx env
      ctx' = applyEnvCtx env ctx
    in
      case checkDuplicates params of
        Just err -> myErr err
        _ -> M.insert const (ctx', params) env


ctxptps :: GCCtx -> Set Ptp
ctxptps = \c ->
  case c of
    Parx ctxs -> L.foldr S.union S.empty (L.map ctxptps ctxs)
    Brax p brc -> L.foldr S.union S.empty (L.map ctxptps (M.elems brc))
    Seqx ctxs -> L.foldr S.union S.empty (L.map ctxptps ctxs)
    Repx ptp ctx -> S.insert ptp (ctxptps ctx)
    Actx (s,r) m -> S.fromList [s,r]
    _ -> S.empty


fill :: GCEnv -> GC -> GCCtx -> GC
fill env g ctx =
{- 
  PRE:  g and ctx are constant-free

  POST: return the g-choreography obtained by replacing the holes in ctx with g
-}
-- replace the holes in ctx with gc
  case ctx of
    Hole -> g
    Dox const _ _ -> myErr ("???" ++ "impossible invocation of " ++ const)
    Parx ctxs -> Par (L.map (fill env g) ctxs)
    Brax p ctxs -> Bra p (M.map (fill env g) ctxs)
    Seqx ctxs -> Seq (L.map (fill env g) ctxs)
    Repx ptp ctx' -> Rep ptp (fill env g ctx')
    Actx c m -> Act c m
    Empx -> Emp

fillx :: GCEnv -> GC -> GCCtx -> GCCtx
fillx env g ctx =
  case ctx of
    Hole -> contexify g
    Dox const _ _ -> myErr ("???" ++ "impossible invocation of " ++ const)
    Parx ctxs -> Parx (L.map (fillx env g) ctxs)
    Brax p ctxs -> Brax p (M.map (fillx env g) ctxs)
    Seqx ctxs -> Seqx (L.map (fillx env g) ctxs)
    Repx ptp ctx' -> Repx ptp (fillx env g ctx')
    _ -> ctx

contexify :: GC -> GCCtx
contexify g =
  case g of
    Par gcs -> Parx (L.map contexify gcs)
    Bra p brc -> Brax p (M.map contexify brc)
    Seq gcs -> Seqx (L.map contexify gcs)
    Rep ptp gc' -> Repx ptp (contexify gc')
    Act c m -> Actx c m
    Emp -> Empx

synErr :: Int -> Int -> Token -> String
synErr l c token =
  "Syntax error at <" ++
	(show (l+1)) ++ "," ++
	(show $ c+1) ++ ">: " ++
	err
  where
    err =
      case token of
        TokenStr s  ->  "unexpected or malformed string: \'" ++ s ++ "\'\n\t characters in " ++ (show separators) ++ " are forbidden"
        TokenQStr s ->  "unexpected or malformed qos string: \'" ++ s ++ "\'"
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
        TokenMAr    ->  "unexpected =>"
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


myErr :: String -> a
myErr err = error ("gcparser: ERROR - " ++ err)


checkDuplicates :: [String] -> Maybe String
checkDuplicates l =
  case l of
    [] -> Nothing
    p:l' ->
      if (p € l')
      then Just ("Duplicated parameter: " ++ (show p))
      else checkDuplicates l'


rename :: M.Map String String -> String -> String
rename m x =
  if x € (M.keys m)
  then m M.! x
  else x


ptpsBranches :: [((GC, Set Ptp), ReversionGuard)] -> Set Ptp
-- to be revised: also participants in constants to be taken
ptpsBranches =
  \l -> L.foldr S.union S.empty (L.map (snd . fst) l)

checkGuard :: (GC, Set Ptp) -> ReversionGuard -> ((GC, Set Ptp), ReversionGuard)
checkGuard gc@(g, ptps) m =
  let
    tmp = [ x | x <- M.keys m, not (S.member x ptps) ]
  in
    if L.null tmp
    then (gc, m)
    else myErr ("Unknown participant" ++ (if L.length tmp > 1 then "(s): " else "") ++ (mkSep tmp ", "))

checkToken :: Token -> GC -> [GC]
checkToken t g =
-- flatten parallel and sequential composition
  case t of
    TokenPar -> case g of
      Par l -> L.foldr (++) [] (L.map (checkToken t) l)
      _ -> [g]
    TokenBra -> case g of
      Bra _ l -> L.foldr (++) [] (L.map (checkToken t) (M.elems l))
      _ -> [g]
    TokenSeq -> case g of
      Seq l -> L.foldr (++) [] (L.map (checkToken t) l)
      _ -> [g]
    _        -> [g]

checkTokenx :: Token -> GCCtx -> [GCCtx]
checkTokenx t ctx =
  case t of
    TokenPar -> case ctx of
      Parx l -> L.foldr (++) [] (L.map (checkTokenx t) l)
      _ -> [ctx]
    TokenBra -> case ctx of
      Brax _ l -> L.foldr (++) [] (L.map (checkTokenx t) (M.elems l))
      _ -> [ctx]
    TokenSeq -> case ctx of
      Seqx l -> L.foldr (++) [] (L.map (checkTokenx t) l)
      _ -> [ctx]
    _        -> [ctx]

}


{-
  A very basic grammar and parser for the textual editing of global
  graphs. The grammar is a revised version of the one used in the
  ICE16 paper with the extensions for reversibility-enabled graphs of
  DAIS 18

     G ::= (o)
        |  P -> P : M
        |  P => P, ..., P : M
	|  G | G
        |  sel { Brc }
        |  sel P { Brc }
        |  branch { Brc }
        |  branch P { Brc }
        |  G ; G
        |  * G @ P
        |  repeat { G unless guard }
        |  repeat P { G unless guard }
        |  { G }

     Brc   ::= G | G unless guard | B + B

     guard ::= P % str | P % str guard

  where '(o)' has a special role: it marks a point where the selector
  of a loop may exit the iteration. Guards are used only for the
  reversible semantics and the string in them is supposed to be some
  valid Erlang code. Likewise for the 'sel' construct, which
  generalises the choice for the reversible semantics. Notice that
  the 'sel' and the 'branch' constructs have the same semantics and
  allow to specify the selector of the branch (to simplify the
  realisation of projections on Erlang, the selector is mandatory for
  REGs and optional otherwise). The clause 'unless guard' is optional
  in branching and iteration.

  The parser assumes the following equalities

     * G @ P = repeat P {G unless g}
     G_1 + ... + G_n       = sel P { G_1 unless g_1 + ... + G_n unless g_n }
	for all guards g_1, ..., g_n 

  The binary operators _ | _, _ + _, and _ ; _ are given in ascending order of
  precedence.

  Note: strings are made of the following characters

     0123456789
     \_$#&~
     ABCDEFGHIJKLMNOPQRSTUVWXYZ
     abcdefghijklmnopqrstuvwxyz

  and must start with a letter when specifying the identity of a
  participant.

  Reserved characters not allowed in strings are:

     @ . , ; : [ ] { } | + * ! ? - % §

  NOTE: The syntax '( G )' is no longer supported.

  Text enclosed by '[[' and ']]' is treated as multi-line comment
  and, after '..', so is the rest of a line.

  Basic syntactic checks are made during the parsing (e.g, (i) that
  sender and receiver of interactions have to be different and (2)
  that the participant controlling a loop is active in the loop). More
  checks are planned together with improved error handling.


  Extensions

  - G-choreography contexts are introduced with the following syntax:

       GCCtx ::= []
            |  (o)
            |  P -> P : M
            |  P => P ... P : M
            |  do str
            |  choiceop { GCCtx }
            |  choiceop { GCCtx }
            |  GCCtx ; GCCtx
            |  repeat { [] }
            |  * { GCCtx } @ ptp
            |  { GCCtx }

    Contexts are instrumental to introduce a rudimentary mechanism of
    definition of g-choreography constants and contexts is provided
    by the following syntactic construct

       let X_1 = Ctx_1 & ... & X_n = Ctx_n in G_1 | ... | G_m

    where X_1, ..., X_n are pairwise different and the set of mappings
    should not yield recursive definitions. To use a constant the
    productions g-choreographies are extended as follows:

       G ::= ...
          |  do X
          |  do X[G]

    where 'do X = do X[(o)].

    The parser flags uses in g-choreography of undefined constants;
    likewise, the parser checks that Ctx_i uses only constants X_j
    with j < i.
-}
