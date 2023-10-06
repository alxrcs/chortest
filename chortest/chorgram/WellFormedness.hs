--
-- Author: Emilio Tuosto <emilio.tuosto@gssi.it>
--
{- This module implements the well-formedness checking.  Most of the
   implementation is adapted from and complements PomCho's closure
   conditions. Some checking consists of syntactic analysis, while
   other conditions require to analyse the pomset semantics of
   g-choreographies.
-}

module WellFormedness where

import Misc
import CFSM (Ptp, isSend, isReceive, senderOf, receiverOf, messageOf)
import SyntacticGlobalChoreographies
import PomsetSemantics
import Data.Set as S
import Data.List as L
import Data.Map.Strict as M

data Role = Active | Passive
  deriving (Ord, Eq)

interactionsOf :: GC -> Set GC
interactionsOf gc =
-- returns the set of interactions occurring in gc
  case gc of
    Emp -> S.empty
    Act (s,r) m -> S.singleton (Act (s,r) m)
    Par gs -> S.unions (S.fromList $ L.map interactionsOf gs)
    Bra _ gs -> S.unions $ M.elems $ (M.map interactionsOf gs)
    Seq gs -> S.unions (S.fromList $ L.map interactionsOf gs)
    Rep _ gc' -> interactionsOf gc'

-- Extracting information from an interaction
sender :: GC -> Ptp
sender (Act (s, _) _) = s
sender gc = error $ "\'sender\' applied to " ++ (show gc)

receiver :: GC -> Ptp
receiver (Act (_, r) _) = r
receiver gc = error $ "\'receiver\' applied to " ++ (show gc)

filterPtp :: Ptp -> GC -> GC
filterPtp p gc =
  -- projects gc on p...without splitting interactions
  case gc of
    Emp -> Emp
    Act (s,r) _ ->
      if p == s || p == r
      then gc
      else Emp
    Par gs -> Par (L.map (filterPtp p) gs)
    Bra sel gs -> Bra sel (M.map (filterPtp p) gs)
    Seq gs -> Seq (L.map (filterPtp p) gs)
    Rep p' gc' -> Rep p' (filterPtp p gc')

firstOnly :: GC -> GC
firstOnly gc =
  -- returns a g-choreography with the same "top-level" structure of
  -- gc containing "first" interations only.
  -- For example, on A -> B: m | ((o) ; C -> D: n) the result is
  -- A -> B: m | C -> D: n
  case gc of
    Emp -> gc
    Act (_, _) _ -> gc
    Par gc -> Par (L.map firstOnly gc)
    Bra p gc -> Bra p (M.map firstOnly gc)
    Seq gc ->
      let
        gc' = L.filter (\x -> not. isEmp $ simplifyGC x) gc
      in
        case gc' of
          [] -> Emp
          _ -> firstOnly $ head gc'
    Rep p gc' -> Rep p (firstOnly $ simplifyGC gc')

lastOnly :: GC -> GC
lastOnly gc =
-- returns a g-choreography with the same "bottom-level"
-- structure of gc with the "last" interations
-- eg if gc = (o) | (G ; C -> D: n ; (o)) then
-- lastOnly gc = (o) | C -> D: n 
  case gc of
    Emp -> gc
    Act (_, _) _ -> gc
    Par gc -> Par (L.map lastOnly gc)
    Bra p gc -> Bra p (M.map lastOnly gc)
    Seq gc ->
      let
        gc' = L.filter (\x -> not. isEmp $ simplifyGC x) gc
      in
        case gc' of
          [] -> Emp
          _ -> lastOnly $ L.last gc'
    Rep p gc' -> Rep p (lastOnly gc')

naiveWB :: Role -> Map Label GC -> Ptp -> Bool
naiveWB Active branching p =
{-
   PRE:  M.size branching > 1    AND   all g-choreography in 'branching' are simplified
   POST: checks if p is active or passive in 'branching'
-}
  -- TODO: return lists of problematic branches when the condition is violated
  let
    pBra = M.map (simplifyGC . firstOnly . (filterPtp p)) branching
    gcs = M.elems pBra
    emptyness = L.all (not . isEmp) gcs
    ints = M.map interactionsOf pBra
    outputOnly = L.all (\a -> L.all (\x -> isAct x && p == (sender x)) a) ints
    disjointness =
      case pairwiseDisjoint ints of
        Nothing -> True
        _ -> False
  in emptyness && outputOnly && disjointness

naiveWB Passive branching p =
  let
    pBra = M.map (simplifyGC . firstOnly . (filterPtp p)) branching
    gcs = M.elems pBra
    emptyness = (L.all isEmp gcs) || (L.all (not . isEmp) gcs)
    ints = M.map interactionsOf pBra
    filterInputs = \a -> S.foldl (\b x -> b && isAct x && (p == (receiver x))) True a
    inputOnly = L.all filterInputs ints
    disjointness =
      case pairwiseDisjoint ints of
        Nothing -> True
        _ -> False
  in emptyness && inputOnly && disjointness

cutFirst :: GC -> (Set GC, GC)
cutFirst gc =
-- separates the first interactions of gc from the rest
  case simplifyGC gc of
    Emp -> (S.empty, Emp)
    Act (_, _) _ -> (S.singleton $ simplifyGC gc, Emp)
    Par gs ->
      let tmp = L.map cutFirst gs
      in (S.unions (L.map fst tmp), Par (L.map snd tmp))
    Bra p gs ->
      let tmp = M.map cutFirst gs
      in (S.unions (S.map fst (S.fromList $ M.elems tmp)), Bra p (M.map snd tmp))
    Seq gs ->
      let (f,r) = cutFirst (L.head gs)
      in (f, Seq (r:(L.tail gs)))
    Rep p gc' ->
      let (f,r) = cutFirst gc'
      in (f, Rep p r)

dependency :: (Set Ptp, Set Ptp) -> GC -> (Set Ptp, Set Ptp)
dependency (min_int, oth) gc =
{-
   computes the set of senders of minimal interactions and those
   participants whose actions causally depend on actions of the ones
   in min_int
-}
  case gc of
    Emp -> (min_int, oth)
    Act (s, r) _ ->
      if S.member s oth
      then (min_int, S.insert r oth)
      else (S.insert s min_int, S.insert r oth)
    Par gs ->
      let tmp = L.map (dependency (min_int, oth)) gs
      in (S.unions $ L.map fst tmp, S.unions $ L.map snd tmp)
    Bra _ gs -> 
      let tmp = M.map (dependency (min_int, oth)) gs
      in (S.unions $ S.map fst (S.fromList $ M.elems tmp),
          S.unions $ S.map snd (S.fromList $ M.elems tmp))
    Seq gs -> L.foldl (\(m,o) g -> dependency (m,o) g) (min_int, oth) gs
    Rep p gc' ->
      if S.member p oth
      then dependency (min_int, oth) gc'
      else dependency (S.insert p min_int, oth) gc'

excerpt :: GC -> String
excerpt gc = "\n-------------" ++
  (L.take 100 $ gc2txt 0 gc) ++
  "..."

ws :: GC -> Maybe String
ws gc =
{-
   checks for well-sequencedness and returns a g-choreography that
   cannot be put in sequence with its subsequent g-choreography (if
   any).
-}
  case gc of
    Emp -> Nothing
    Act _ _ -> Nothing
    Par gcs ->
      let
        tmp = L.foldr aux "" (L.map ws gcs)
      in
        if tmp == ""
        then Nothing
        else Just (tmp ++ (excerpt gc))
    Bra _ gcs ->
      let
        tmp = L.foldr aux "" (L.map ws (M.elems gcs))
      in
        if tmp == ""
        then Nothing
        else Just (tmp ++ (excerpt gc))
    Seq gcs ->
      case gcs of
        [] -> Nothing
        gc:gcs' ->
          case ws gc of
            Nothing ->
              if gcs' == []
              then Nothing
              else
                case wsAux gc (head gcs') of
                  Nothing -> ws (Seq gcs')
                  _ -> wsAux gc (head gcs')
            _ -> ws gc
--          (ws gc') && (wsAux gc' gc'') && (ws (Seq (tail gcs)))
    Rep _ gc' -> ws gc'
  where
    aux x acc =
      case x of
        Nothing -> ""
        Just s -> acc ++ "\n"


wsAux :: GC -> GC -> Maybe String
wsAux gc gc' =
  -- NOTE: the implementation differs from the formal definition
  -- because the check that minimal imputs of gc' depend on some
  -- output of gc is done syntactically instead of using pomsets.
  --
  -- (isEmp $ simplifyGC gc) ||
  -- (isEmp $ simplifyGC gc') ||
  -- (inCond && L.all outCond ps')
  if (isEmp $ simplifyGC gc) || (isEmp $ simplifyGC gc')
  then Nothing
  else
    let
      (sem, e') = pomsetsOf gc 1 0
      (sem', _) = pomsetsOf gc' 1 (e'+1)
      (ps, ps') = (S.toList $ sem, S.toList $ sem')
      ptpgc = gcptps gc
      inCond =
        let
          ptps = gcptps gc
          min = interactionsOf $ firstOnly gc'
          aux (Act (s,r) _ ) =
            (S.member r ptps) || (S.member s ptps)
        in
          S.filter (not . aux) min
      outCond r' =
        let
          l' = labelOf r'
          outs' = S.filter (\e -> isSend (l'!e) && S.member (senderOf (l'!e)) ptpgc) (eventsOf r')
          chk r =
            let
              l = (labelOf r)
              outs = S.filter (\e -> isSend (l!e)) (eventsOf r)
              mkAction ((s,r), _, m) = Act (s,r) m
              pairs =
                (S.fromList [(e,e') | e <- S.toList outs, e' <- S.toList outs',
                            S.member (mkAction (l'!e')) (interactionsOf gc)
                            ]
                )
                S.\\
                (orderOf $ seqLeq (r, r'))
            in
              S.map (\(e1,e2) -> ((labelOf r)!e1,(labelOf r')!e2)) pairs
        in
          S.filter (not . S.null) (S.map chk sem)
    in
      let
        prtAction a@((s,r), d, m) =
          if isSend a
          then s ++ " -> " ++ r ++ ": " ++ m
          else
            error ("Unexpected action " ++ (show a))
        prtGC g =
          case g of
            Act (s,r) m ->
              s ++ " -> " ++ r ++ ": " ++ m
            _ -> "Interaction expected" 
        prtPair = \(a1,a2) -> (prtAction a1) ++
          " in parallel with " ++ (prtAction a2)
        aux =
          S.foldr (\a b -> (prtGC a) ++ "\n" ++ b) ""
        tmpOut =
          (S.unions $ S.unions $ L.map outCond ps')
        aux' =
          L.foldr (\a b -> (prtPair a) ++ b) ""
      in
        case (S.null inCond, S.null tmpOut) of
          (True, True) -> Nothing
          (False, True) -> Just (" on the input in " ++ (aux inCond) ++ (excerpt (Seq [gc, gc'])))
          (True, False) -> Just (" on the output in " ++ (aux' tmpOut) ++ (excerpt (Seq [gc, gc'])))
          (False, False) -> Just ((" on the input in " ++ (aux inCond) ++ (excerpt (Seq [gc, gc'])))
                                  ++ (" on the output in " ++ (aux' tmpOut)) ++ (excerpt (Seq [gc, gc'])))
        
wf :: GC -> Maybe (GC, GC, String)
wf gc =
-- check well-forkedness and return the set of interactions occurring
-- on more than one thread
  case gc of
    Emp -> Nothing
    Act (_,_) _ -> Nothing
    Par gs ->
      let f = M.fromList $ (L.zip (range $ L.length gs) (L.map interactionsOf gs))
      in
        case pairwiseDisjoint f of
          Nothing -> Nothing
          Just (i,j) -> Just (gs!!i, gs!!j, excerpt gc)
    Bra p gcs ->
      case M.elems gcs of
        [] -> Nothing
        g:gs' ->
          if wf g == Nothing
          then wf (Bra p (M.fromList $ L.zip ([1 .. L.length gs']) gs'))
          else wf g
    Seq gs ->
      case gs of
        [] -> Nothing
        g:l ->
          if wf g == Nothing
          then wf (Seq l)
          else wf g
    Rep _ gc' -> wf gc'

wb :: GC -> Maybe String
wb gc =
{-
   check well-branchedness
-}
  case gc of
    Emp -> Nothing
    Act (_,_) _ -> Nothing
    Par gs ->
      let
        err = L.find notNothing (L.map wb gs)
      in
        case err of
          Nothing -> Nothing
          Just e -> e
    Bra p gcs ->
      let ptps = gcptps gc
          gcs' = M.map simplifyGC gcs
          getActive = S.filter (naiveWB Active gcs') ptps
          getPassive = S.filter (naiveWB Passive gcs') ptps
          mkList = \x y -> x ++ " " ++ y
          activeFlag = p /= "" && not(S.member p getActive)
          passiveFlag = (S.size getPassive) /= (S.size ptps) - 1
        in
          case S.size getActive of
            0 ->
              if L.all (== (L.head $ M.elems gcs)) (L.tail $ M.elems gcs)
              then Nothing
              else Just ("No active participant" ++ (excerpt gc))
            1 ->
              if passiveFlag || activeFlag
              then Just (
                (if activeFlag
                 then "Participant " ++ p ++ " is not active (as declared)"
                 else "") ++
                (if passiveFlag
                  then
                   (if activeFlag then " and\n" else "") ++
                   "Either some non-active participant is not passive," ++
                   "\n\tor no active or passive participants:" ++
                   "\n\t\tactive: " ++ (S.foldr mkList "" getActive) ++
                   "\n\t\tpassive: " ++ (S.foldr mkList "" getPassive)
                  else ""
                ) ++ (excerpt gc)
                )
              else
                if L.all (== (L.head $ M.elems gcs)) (L.tail $ M.elems gcs)
                then Nothing
                else
                  let
                    pgc p = M.elems $ M.map (simplifyGC . filterPtp p ) gcs'
                    aux p = L.all (== (L.head $ (pgc p))) (L.tail $ (pgc p))                            
                    uniform = S.filter aux getActive
                  in
                    case S.toList (getActive S.\\ uniform) of
                      [] -> Nothing
                      [_] -> Nothing
                      _ -> Just ("There are several active participants: " ++
                                 (S.foldr mkList "" (uniform S.\\ getActive)) ++
                                 (excerpt gc)
                                )
            _ -> Just ("There are several active participants: " ++
                       (S.foldr mkList "" getActive) ++
                       (excerpt gc)
                      )
    Seq gs -> 
      let
        err = L.find notNothing (L.map wb gs)
      in
        case err of
          Nothing -> Nothing
          Just e -> e
    Rep _ gc' -> wb gc'

