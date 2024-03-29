--
-- Author: Emilio Tuosto <emilio.tuosto@gssi.it>
--

{-
	Project a syntactic g-choreography, calculate
	its semantics, and output it in dot format.
-}

module SyntacticGlobalChoreographies where

import Data.Set as S
import Data.List as L
import Data.Map.Strict as M
import Misc
import CFSM
import DotStuff
import Data.String.Utils (replace)

-- Simple internal representation of labels of branches
type Label = Int

{-
   A syntactic global graph is a set of nodes, a source, a sink, and a
   set of edges We assume that cp's will be automatically generated
   (uniquely) during parsing
-}
data GC = Emp
        | Act Channel Message
        | Par [GC]
        | Bra Ptp (Map Label GC)
        | Seq [GC]
        | Rep Ptp GC
    deriving (Eq, Ord, Show)

-- Basic functions to check the nature of a g-choreography
isAct :: GC -> Bool
isAct (Act _ _) = True
isAct _ = False

isBra :: GC -> Bool
isBra (Bra _ _) = True
isBra _ = False

isEmp :: GC -> Bool
isEmp Emp = True
isEmp _ = False


-- Other auxiliary operations

gcptps :: GC -> Set Ptp
gcptps gc =
-- return the set of participants of gc
  case gc of
    Emp -> S.empty
    Act (s,r) _  -> S.fromList [s,r]
    Par gs -> L.foldr S.union S.empty (L.map gcptps gs)
    Bra p gs -> L.foldr S.union S.empty (L.map gcptps (M.elems gs))
    Seq gs -> L.foldr S.union S.empty (L.map gcptps gs)
    Rep p g' -> S.insert p (gcptps g')

loopBack :: State -> State
loopBack q = "__l__" ++ q

loopExit :: State -> State
loopExit q = "__e__" ++ q

simplifyGC :: GC -> GC
simplifyGC gc =
{-
  simplifies gc by flattening nested | and + according to the
  following structural congruence rules are:

	(o) + gc = gc
	( GC, _|_, (o) ) abelian monoid
	( GC, _;_, (o) ) monoid
-}
  case gc of
    Emp -> Emp
    Act (_, _) _ -> gc
    Seq gcs ->
      let
        gcs' = [g | g <- (L.map simplifyGC gcs), g /= Emp]
      in
        case gcs' of
          [] -> Emp
          [gc'] -> gc'
          _ -> Seq gcs'
    Par gcs ->
      let
        gcs' = [g | g <- (L.map simplifyGC gcs), g /= Emp]
      in
        case gcs' of
          [] -> Emp
          [gc'] -> gc'
          _ -> Par gcs'
    Bra sel gcs ->
      let
        shift k m =
          M.fromList [(i+k,v) | (i,v) <- M.toList m]
        (bra, oth) =
          L.partition isBra (M.elems $ M.map simplifyGC gcs)
        ms =
          [m | Bra sel m <- bra]
        aux =
          M.fromList $ L.zip ([1 .. (L.length oth)]) oth
        gcs' =
          L.foldl (\x y -> M.union x (shift (M.size x) y)) aux ms
      in
        if (S.size $ S.fromList $ M.elems gcs') == 1 
        then L.head $ M.elems gcs'
        else Bra sel gcs'
    Rep p gc' ->
      let body = simplifyGC gc'
      in
        case body of
          Emp -> Emp
          _ -> Rep p body

unfoldGC :: Int -> GC -> GC
unfoldGC n gc =
{-
   PRE: n >= 0
   POST: each loop in gc is unfolded n times
-}
    case gc of
      Emp -> gc
      Act (s,r) m -> gc
      Par gcs -> Par (L.map (unfoldGC n) gcs)
      Bra p gcs -> Bra p (M.map (unfoldGC n) gcs)
      Seq gcs -> Seq (L.map (unfoldGC n) gcs)
      Rep _ g -> Seq ([unfoldGC n g | x <- range(n)])
  

proj :: GC -> State -> State -> Ptp -> Int -> P -> (CFSM, State)
proj gc q0 qe p n pmap =
  {- 
     PRE:
	- actions are well formed (wffActions) ^ q0 /= qe
	- p is a participant of gc
	- the participants of gc are all and only those in M.elems pmap
	- n controls the projection of loops:
          * n >= 0 is the number of unfoldings
          * n < 0 use normal projections without unfoldings
     POST:
	the non-minimised projection of gc wrt p and a unique
        interface state (it must always exist!)  q0 must be the
        initial state of the resulting cfsm and qe is the interface
        state, respectively
  -}
  let
    g_loop = \s ptps ->
      Seq (L.map (\r -> Act (s,r) (loopBack q0)) (S.toList $ S.delete s ptps))
    g_exit = \s ptps ->
      Seq (L.map (\r -> Act (s,r) (loopExit qe)) (S.toList $ S.delete s ptps))
    onetr = \q1 q2 l ->
      (((S.fromList [q1, q2]), q1, S.singleton l, (S.singleton (q1, l, q2))), q2)
    inverse =
      inv pmap
    pTau = \l ->
      ((show $ inverse!p, show $ inverse!p), l, "")
    dm = \q_ l ->
      onetr q0 q_ (pTau l)
  in
    case gc of
      Emp ->
        dm qe Tau
      Act (s,r) m ->
        case (p==s,p==r) of
          (False, False) -> dm qe Tau
          _ -> onetr q0 qe label
            where
              label =
                if p==s
                then ((show $ inverse!p, show $ inverse!r), Send, m)
                else ((show $ inverse!s, show $ inverse!p), Receive, m)
      Par gcs ->
        (replaceState (initialOf m) q0 m, qe)
        where
          m   = replaceState qe' qe (cfsmProd $ L.map fst mps)
          qe' = L.foldr stateProd "" (L.map snd mps)
          mps = L.map (\g -> proj g q0 qe p n pmap) gcs
      Bra _ gcs ->
        (replaceStates (\q_ -> q_ € [qe ++ (show i) | i <- [1 .. (length mps)]]) qe cfsm, qe)
        where
          (states, acts, trxs) =
            L.foldl
              (\(x,y,z) m -> (S.union x (statesOf m),
                              S.union y (actionsOf m),
                              S.union z (transitionsOf m)
                             )
              )
            (S.singleton qe, S.singleton $ pTau Tau, S.empty)
            (L.map fst mps)
            -- gcs' = L.zip (S.toList gcs) [1 .. S.size gcs]
          mps  =
            L.map (\(i,g) -> proj g q0 (qe ++ (show i)) p n pmap) (M.toList gcs)
          cfsm =
            replaceStates (\q_ -> q_ € [q0 ++ (show i) | i <- [1 .. (length mps)]]) q0 (states, q0, acts, trxs)
      Seq gcs ->
        ( replaceState qe' qe (states, q0, acts, trxs) , qe )
        where
          (_, qe', states, acts, trxs) =
            L.foldl
              (\(i, qi, x, y, z) g ->
                    let (m, qf') = proj g qi (qe ++ (show i)) p n pmap
                    in
                     (i + 1,
                      qf',
                      S.union x (statesOf m),
                      S.union y (actionsOf m),
                      S.union z (transitionsOf m)
                     )
                  )
              (0, q0, S.empty, S.empty, S.empty)
              gcs
      Rep p' g ->
        if (S.member p ptpsloop)
        then (fst m, qe)
        else (dm qe Tau)
        where
          ptpsloop = gcptps g
          suf = if n<0 then show (-n) else show n
          m =
            if n >= 0
            then
              case (p == p' , n == 0) of
                (True, True) ->
                  proj (g_exit p ptpsloop) q0 qe p n pmap
                (True, False) ->
                  let
                    (cfsm_exit, _) =
                      proj (g_exit p ptpsloop) q0 qe p (n-1) pmap
                    (cfsm_loop, qel) =
                      proj (g_loop p ptpsloop) q0 (qe ++ suf) p (n-1) pmap
                    (cfsm_body, qeb) =
                      proj g qel (qe ++ "body" ++ suf) p (n-1) pmap
                    (cfsm_itr, qu) =
                      proj (Rep p g) qeb (qe ++ "iter" ++ suf) p (n-1) pmap
                    cfsm_bra =
                      cfsmUnion q0 [cfsm_exit, cfsm_loop]
                    cfsm =
                      replaceState (qe ++ "iter" ++ suf) qe (cfsmUnion q0 [cfsm_bra, cfsm_body, cfsm_itr])
                  in
                    (cfsm, qu)
                (_, _) ->
                  let
                    (cfsm_loop, qel) =
                      onetr q0 (qe ++ suf) ((show $ inverse!p', show $ inverse!p), Receive, (loopBack q0))
                    (cfsm_exit, _) =
                      onetr q0 qe ((show $ inverse!p', show $ inverse!p), Receive, (loopExit qe))
                    (cfsm_body, _) =
                      proj g qel q0 p (n-1) pmap
                    cfsm =
                      cfsmUnion q0 [cfsm_loop, cfsm_exit, cfsm_body]
                  in
                    (cfsm, qe)
            else
              if p == p'
              then
                let
                  (cfsm_loop, qel) =
                    proj (g_loop p ptpsloop) q0 (qe ++ suf) p n pmap
                  (cfsm_exit, _) =
                    proj (g_exit p ptpsloop) q0 qe p n pmap
                  (cfsm_body, _) =
                    proj g qel q0 p n pmap
                  cfsm =
                    cfsmUnion q0 [cfsm_loop, cfsm_exit, cfsm_body]
                in
                  (cfsm, qe)
              else
                let
                  (cfsm_loop, qel) =
                    onetr q0 (qe ++ suf) ((show $ inverse!p', show $ inverse!p), Receive, (loopBack q0))
                  (cfsm_exit, _) =
                    onetr q0 qe ((show $ inverse!p', show $ inverse!p), Receive, (loopExit qe))
                  (cfsm_body, _) =
                    proj g qel q0 p (n-1) pmap
                  cfsm =
                    cfsmUnion q0 [cfsm_loop, cfsm_exit, cfsm_body]
                in
                  (cfsm, qe)

projx :: Bool -> GC -> P -> Ptp -> State -> State -> Int -> (CFSM, State)
projx loopFlag gc pmap p q0 qe n =
--
-- PRE : actions are well formed (wffActions) ^ q0 /= qe ^ p is a participant of gc
-- POST: the non-minimised projection of gc wrt p and a unique exiting state (it must always exist!)
--       n is a counter for fresh state generation
--       q0 and qe correspond to the entry and exit state, respectively
--
-- Parameter 'loopFlag' is set to true when projecting loops
-- and to false otherwise.
--
  let inverse = inv pmap
      taul l = ((show $ inverse!p, show $ inverse!p), l, "")
      tautrx = \q1 q2 l ->  S.singleton (q1, taul l, q2)
      dm q l = ((S.fromList [q0, q], q0, S.singleton (taul l), tautrx q0 q l), q)
  in case gc of
      Emp ->
        if loopFlag
        then (dm (qe ++ "Break") BreakLoop)
        else (dm qe Tau)
      Act (s,r) m ->
        if (p /= s && p /= r)
        then dm qe Tau
        else (((S.fromList [q0, qe]), q0, S.singleton c, (S.singleton (q0, c, qe))), qe)
        where c = if (p == s)
                  then ((show $ inverse!p, show $ inverse!r), Send, m)
                  else ((show $ inverse!s, show $ inverse!p), Receive, m)
      Par gcs -> (replaceState (initialOf m) q0 (S.insert qe (statesOf m),
                                                 initialOf m,
                                                 S.insert (taul Tau) (actionsOf m),
                                                 (transitionsOf m)
                                                ),
                   qe
                 )
        where m   = replaceState qe' qe (cfsmProd $ L.map fst mps)
              qe' = L.foldr stateProd "" (L.map snd mps)
              mps = L.map (\g -> projx loopFlag g pmap p q0 qe n) gcs
      Bra _ gcs     -> (replaceStates (\q -> q € [qe ++ (show i) | i <- [1 .. (length mps)]]) qe (replaceStates (\q -> q € [q0 ++ (show i) | i <- [1 .. (length mps)]]) q0 (states, q0, acts, trxs)), qe)
        where (states, acts, trxs) = L.foldl
                (\(x,y,z) m -> (S.union x (statesOf m),
                                S.union y (actionsOf m),
                                S.union z (transitionsOf m) )
                )
                (S.singleton qe, S.singleton $ taul Tau, S.empty)
                (L.map fst mps)
              -- gcs' = L.zip (S.toList gcs) [1 .. S.size gcs]
              mps  = L.map (\(i,g) -> projx loopFlag g pmap p q0 qe n) (M.toList gcs)
      Seq gcs -> ( replaceState qe' qe (states, q0, acts, trxs) , qe )
        where (_, qe', states, acts, trxs) =
                L.foldl
                  (\(i, qi, x, y, z) g ->
                    let (m, qf') = projx loopFlag g pmap p qi (qe ++ (show i)) n in
                     (i + 1,
                      qf',
                      S.union x (statesOf m),
                      S.union y (actionsOf m),
                      S.union z (transitionsOf m)
                     )
                  )
                  (0, q0, S.empty, S.empty, S.empty)
                  gcs
      Rep p' g -> if (S.member p bodyptps) then (gcrep, qe') else (dm qe' Tau)
        where bodyptps    = gcptps g
              gcrep       = (S.unions [statesOf body, statesOf loop, statesOf exit],
                             initialOf body,
                             S.unions [actionsOf body, actionsOf loop, actionsOf exit],
                             S.unions [transitionsOf body, transitionsOf loop, transitionsOf exit]
                            )
              suf         = show n
              (body', q)  = projx True g pmap p q0 (qe ++ suf) (n + 2)
              breakPoints = S.map (\(_, _, q_) -> q_) (S.filter (\(_, (_, l, _), _) -> l == BreakLoop) (transitionsOf body'))
              body        = replaceStates (\q_ -> q_ € S.toList breakPoints) (qe ++ suf) body'
              (loop', ql) = projx False (helper (lpref ++ suf)) pmap p q (q0 ++ suf) (n + 2)
              loop        = replaceState ql q0 loop'
              (exit, qe') = projx False (helper (epref ++ suf)) pmap p q qe (n + 2)
              helper msg  = Par (L.map (\p'' -> Act (p',p'') msg) (S.toList $ S.delete p' bodyptps))


-- Representing GCs in DOT format

-- The dot graph is a pair made of a list of nodes and a list of edges
type PD = ([(DotNode, DotString)], [(DotNode, DotNode)])

node2dot :: DotNode -> DotString
node2dot n =
-- DOT representation of GC nodes
  (if n < 0 then "_" else "") ++ (show $ abs n)
      
gc2dot :: GC -> String -> Map String String -> DotString
gc2dot gc name flines =
{- 
   transform a GC in dot format giving it name 'name'
   and setting the size of nodes to 'nodeSize'
-}
  let
    nodeSize = flines!gcsizenode
    maxIdx vs = aux vs 0
      where aux [] v = v + 1
            aux ((v', _):vs') v = aux vs' (max v v')
    dummyGC n = ([(n, branchV), (-n, mergeV)], [(n, -n)])
    helper vs as gc_  =
      let
        (sink, i) = (L.last vs, 1 + (maxIdx vs))
        attach idx idx' =
          [(s, t)   | (s, t) <- as, t /= fst sink] ++
          [(s, idx) | (s, t) <- as, t == fst sink] ++
          [(idx', fst sink)]
        notgate = \v -> v /= i && v /= (-i)
      in
        case gc_ of
          Emp ->
            (vs, as)
          Act _ _ ->
            ((L.init vs) ++ [(i, dotLabelOf gc_ )] ++ [sink], attach i i)
          Par gcs ->
            ((L.init vs) ++ vs' ++ [sink], (attach i (-i)) ++ as')
            where
              (vs', as') = unionsPD forkV joinV notgate i (rename (\v -> not (notgate v)) i graphs)
              graphs     = (L.map (helper [(i,forkV),(-i,joinV)] [(i,-i)]) gcs)
          Bra _ gcs  -> ((L.init vs) ++ vs' ++ [sink], (attach i (-i)) ++ as')
            where
              (vs', as') = unionsPD branchV mergeV notgate i (rename (\v -> not (notgate v)) i graphs)
              (evs, eas) = dummyGC i
              -- graphs     = S.toList (S.map (helper evs eas) gcs)
              graphs     = M.elems (M.map (helper evs eas) gcs)
          Seq gcs -> graphy gcs vs as
            where
              graphy gcs_ vs_ as_ =
                case gcs_ of
                  []       -> (vs_,as_)
                  Emp:gcs' -> graphy gcs' vs_ as_
                  gc':gcs' -> graphy gcs' vs'' as''
                    where
                      (vs0,as0)   = helper [(idx,""),(-idx,"")] [(idx,-idx)] gc'
                      (vs',as')   = renameVertex notgate (vs0,as0) (1 + maxIdx vs0)
                      (vs'',as'') = catPD (\(_,l) -> l /= "") (vs_,as_) (vs',as')
                      idx         = 1 + maxIdx vs_
          Rep _ gc' -> ((L.init vs) ++ vs' ++ [sink], (attach i (-i)) ++ ((-i,i):as'))
            where
              (evs, eas) = dummyGC i
              (vs', as') = helper evs eas gc'
        where
          rename excluded offset pds =
            case pds of
              (vs1, as1):pds' -> ([(newNode excluded v offset,l) | (v,l) <- vs1],
                                  [(newNode excluded s offset, newNode excluded t offset) | (s,t) <- as1]) :
                                 (rename excluded (1 + offset + maxIdx vs1) pds')
              []             -> []
          unionsPD gl gl' included idx pds =
            ([(idx,gl)] ++ [(v,l) | (v,l) <- L.concat  $ L.map fst pds, (included v)] ++ [((-idx),gl')],
             L.concatMap snd pds)
    dotnodes vs =
      L.concatMap (\(s, l) -> "\tnode" ++ (node2dot s) ++ l) vs
    dotedges as =
      L.concatMap (\(s, t) -> "\tnode" ++ (node2dot s) ++ " -> node" ++ (node2dot t) ++ "\n") as
    (evs_, eas_) =
      ([(1, sourceV), (-1, sinkV)], [(1, -1)])
    (vertexes, edges) =
      helper evs_ eas_ gc
    header =
      "digraph " ++ name ++
      " {\n   node [width=" ++ nodeSize ++
      ", height=" ++ nodeSize ++ "]\n\n"
  in
    header ++ (dotnodes vertexes) ++ (dotedges edges) ++ "\n}\n"

dotLabelOf :: GC -> DotString
dotLabelOf gc = case gc of
              Emp ->
                sourceV
              Act (s,r) m ->
                " [label = \"" ++ s ++
                " &rarr; " ++ r ++
                " : " ++ m ++
                "\", shape=rectangle,\
                \ fontname=helvetica,\
                \ fontcolor=MidnightBlue]\n"
              Par _
                -> forkV
              Bra _ _  ->
                branchV
              Seq _ ->
                ""
              Rep _ _ ->
                ""

catPD :: ((DotNode, DotString) -> Bool) -> PD -> PD -> PD
catPD included (vs, as) (vs', as') = (vs'', as'')
{-
   catPD (vs,as) (vs',as') appends (vs', as') attaching its
   source to the nodes of (vs,as) entering the sink of (vs,as)
  
   PRE : vs and vs' start and end with the source and target vertex of
         the corresponding graph
  
   POST: the result is the sequential composition of the graphs
         embedding the second just before the sink of (vs,as); the
         source of the resulting graph is the source of (vs,as)
-}  
  where vs'' = (L.init vs) ++ (L.filter included vs') ++ [L.last vs]        
        as'' = [ (n, m) | (n, m) <- as, not((n, m) € maxs) ] ++
               [ (n, m) | (n, _) <- maxs, (_, m) <- min_   ] ++
               [ (n, m) | (n, m) <- as', not((n, m) € (min_ ++ max_)) ] ++
               [ (n, fst $ L.last vs) | (n, _) <- max_  ]
        maxs = maxR as
        min_ = minR as'
        max_ = maxR as'

renameVertex :: (DotNode -> Bool) -> PD -> DotNode -> PD
renameVertex excluded (vs, as) offset =
  ([(newNode excluded s offset, l) | (s,l) <- vs],
   [(newNode excluded s offset, newNode excluded t offset) | (s,t) <- as]
  )


gc2txt :: Int -> GC -> String
gc2txt level gc =
{-
   PRE: 
   POST: pretty printing of g-choreographies
-}
  let
    indent = "  "
    tab l s = "\n" ++ (concat $ replicate l indent) ++ s
  in
    case gc of
      Emp ->
        tab level "(o)"
      Act (s,r) m ->
        tab level (s ++ " -> " ++ r ++ ": " ++ m)
      Par gcs ->
        (tab level ("{")) ++
        intercalate (tab level "|") (L.map (gc2txt (level + 1)) gcs) ++
        (tab level "}")
      Bra p gcs ->
        (tab level ("sel " ++ p ++ " {")) ++
        intercalate (tab (level+1) "+") (L.map (gc2txt (level+1)) (M.elems gcs)) ++
        (tab level "}")
      Seq gcs ->
        intercalate (tab level ";") (L.map (gc2txt level) gcs)
      Rep p g ->
        (tab level ("repeat {")) ++
        p ++ (gc2txt (level + 1) g) ++
        (tab level "}")

--
-- Stuff to generate global graphs in the gml format
--
data GMLTAGS = Source | Sink | Fork | Branch | Join | Merge | Loop
  deriving (Eq, Ord, Show)

gmlstyle :: GMLTAGS -> String
-- using the yed style
gmlstyle tag = ""

gmldata :: String -> String -> String
gmldata k v = "      <data key=\"" ++ k ++ "\">" ++ v ++ "</data>\n"

gmlLabelOf :: GC -> String
gmlLabelOf gc = case gc of
              Act (s,r) m -> (gmldata "sender" s ++ gmldata "receiver" r ++ gmldata "payload" m)
              _       -> ""

gmlNode :: String -> Int -> String
gmlNode d idn  = "    <node id=\"" ++ (show idn) ++ "\">\n" ++ d ++ "    </node>\n"

gmlEdge :: Int -> Int -> String
gmlEdge source target = "    <edge source=\"" ++ (show source) ++ "\" target=\"" ++ (show target) ++ "\"></edge>\n"

gmlOpenGate :: Int -> GMLTAGS -> String
gmlOpenGate idn gate = gmlNode (gmldata "open" (show gate)) idn

gmlCloseGate :: Int -> GMLTAGS -> String
gmlCloseGate idn gate = gmlNode (gmldata "close" (show gate)) (-idn)

gmlrename :: (Int -> Bool) -> Int -> Int -> String -> String
gmlrename excluded n j s =
  if (excluded n)
  then s
  else
    let idn = (if n > 0 then n + j else n - j)
    in replace ("<node id=\"" ++ (show n)) ("<node id=\"" ++ (show idn)) s

gc2graphml :: GC -> String
gc2graphml gc =
--
-- gc2dot gc name transforms a GC in dot format
--
  let
    header = --"<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\" xmlns:java=\"http://www.yworks.com/xml/yfiles-common/1.0/java\" xmlns:sys=\"http://www.yworks.com/xml/yfiles-common/markup/primitives/2.0\" xmlns:x=\"http://www.yworks.com/xml/yfiles-common/markup/2.0\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xmlns:y=\"http://www.yworks.com/xml/graphml\" xmlns:yed=\"http://www.yworks.com/xml/yed/3\" xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlns http://www.yworks.com/xml/schema/graphml/1.1/ygraphml.xsd\">\n"
      "<?xml version=\"1.0\"encoding=\"utf-8\"? standalone=\"no\">\n\
      \<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\"\
      \ xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\
      \ xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlns\
      \ http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd\">\n"
    ogate =
      "  <key attr.name=\"open\" attr.type=\"string\" for=\"node\" id=\"open\" />\n"
    cgate =
      "  <key attr.name=\"close\" attr.type=\"string\" for=\"node\" id=\"close\" />\n"
    sender =
      "  <key attr.name=\"sender\" attr.type=\"string\" for=\"node\" id=\"sender\" />\n"
    receiver =
      "  <key attr.name=\"receive`<r\" attr.type=\"string\" for=\"node\" id=\"receiver\" />\n"
    payload =
      "  <key attr.name=\"payload\" attr.type=\"string\" for=\"node\" id=\"payload\" />\n"
-- TODO: check that cc works as before commenting the next 3 lines
--      source   = ""--  <key attr.name=\"source\" attr.type=\"string\" for=\"node\" id=\"source\" />\n"
--      sink     = ""--"  <key attr.name=\"sink\" attr.type=\"string\" for=\"node\" id=\"sink\" />\n"
--      yattr    = ""--  <key for=\"node\" id=\"ylabel\" yfiles.type=\"nodegraphics\"/>"
    edir =
      "  <graph edgedefault=\"directed\">\n"
    footer =
      "  </graph>\n</graphml>\n"
    maxIdx vs =
      aux vs 0
      where
        aux [] v = v + 1
        aux ((v', _):vs') v = aux vs' (max v v')
    dummyGC n =
      ([(n, gmlOpenGate n Loop),
        (-n, gmlCloseGate n Loop)], [(n, -n)]
      )
    helper vs as gc_  =
      let
        (sink, i) =
          (L.last vs, 1 + (maxIdx vs))
        attach idx idx' =
          [(s, t)   | (s, t) <- as, t /= fst sink] ++
          [(s, idx) | (s, t) <- as, t == fst sink] ++
          [(idx', fst sink)]
        notgate =
          \v -> v /= i && v /= (-i)
      in
        case gc_ of
          Emp ->
            (vs, as)
          Act _ _ ->
            ((L.init vs) ++
               [(i, gmlNode (gmlLabelOf gc_) i)] ++
               [sink],
             attach i i
            )
          Par gcs ->
            let
              (vs', as') =
                gather (gmlOpenGate i Fork)
                (gmlCloseGate i Join)
                notgate
                i
                (rename (not.notgate) i graphs)
              graphs =
                (L.map (helper [(i, (gmlOpenGate i Fork)), ((-i), (gmlCloseGate i Join))] [(i, (-i))]) gcs)
            in
              ((L.init vs) ++ vs' ++ [sink], (attach i (-i)) ++ as')
          Bra _ gcs ->
            let
              gmlOpen =
                gmlOpenGate i Branch
              gmlClose =
                gmlCloseGate i Merge
              (vs', as') =
                gather gmlOpen gmlClose notgate i (rename (not.notgate) i graphs)
                --graphs = S.toList (S.map (helper [(i, (gmlOpenGate i Branch)), ((-i), (gmlCloseGate i Merge))] [(i, (-i))]) gcs)
              graphs =
                M.elems $
                  M.map (helper [(i, gmlOpen), ((-i), gmlClose)] [(i, (-i))]) gcs
            in
              ((L.init vs) ++ vs' ++ [sink], (attach i (-i)) ++ as')
          Seq gcs ->
            let
              graphy gcs_ vs_ as_ =
                case gcs_ of
                  [] -> (vs_,as_)
                  Emp:gcs' -> graphy gcs' vs_ as_
                  gc':gcs' ->
                    let
                      (vs0,as0) =
                        helper [(idx,""),(-idx,"")] [(idx,-idx)] gc'
                      (vs',as') =
                        renameVertex notgate (vs0,as0) (1 + maxIdx vs0)
                      (vs'',as'') =
                        catPD (\(_,l) -> l /= "") (vs_, as_) (vs', as')
                      idx = 1 + maxIdx vs_
                    in
                      graphy gcs' vs'' as''
            in
              graphy gcs vs as
          Rep _ gc' ->
            let
              (evs, eas) = dummyGC i
              (vs', as') = helper evs eas gc'
            in
              ((L.init vs) ++ vs' ++ [sink], (attach i (-i)) ++ ((-i,i):as'))
          where
            rename excluded offset pds =
              case pds of
                [] -> []
                (vs1, as1):pds' ->
                  ([(newNode excluded v offset, gmlrename excluded v offset l) | (v,l) <- vs1],
                   [(newNode excluded s offset, newNode excluded t offset) | (s,t) <- as1]
                  ) : (rename excluded (1 + offset + maxIdx vs1) pds')
            gather gl gl' included idx pds =
              ([(idx,gl)] ++
                 [(v,l) | (v,l) <- L.concat  $ L.map fst pds, (included v)] ++
                 [((-idx),gl')],
               L.concatMap snd pds
              )
    gmlNodes vs =
      L.concatMap (\(_, s) -> s) vs
    gmlEdges as =
      L.concatMap (\(s, t) -> gmlEdge s t) as
    (evs_, eas_) =
      ([(1, (gmlOpenGate 1 Source)), (-1, gmlCloseGate 1 Sink)], [(1, -1)])
    (vertexes, edges) =
      helper evs_ eas_ gc
  in header ++
     ogate ++
     cgate ++
     sender ++
     receiver ++
     payload ++
--     source ++
--     sink ++
--     yattr ++
     edir ++
     (gmlNodes vertexes) ++
     (gmlEdges edges) ++
     footer
