--
-- Authors: Emilio Tuosto <emilio.tuosto@gssi.it>
--
-- Variuos utility functions and definitions
--

module Misc where

import Data.List as L
import Data.Set as S
import Data.Map.Strict as M
import System.FilePath.Posix
import qualified Data.Text as T

type Message             = String
type Edge vertex label = (vertex, label, vertex)
type Graph vertex label = (Set vertex, vertex, Set label, Set(Edge vertex label))

data Command =
  GMC
  | BGG
  | CG
  | GC
  | GC2DOT
  | GC2FSA
  | PROJ
  | UNFOLD
  | PTPS
  | GC2POM
  | TSOF
  | POM2GC
  | GC2GML
  | SYS
  | MIN
  | WB
  | WS
  | WF
  | GC2LATEX
  | CFSM2TIKZ
  | TEST
  | HGSEM -- deprecated
-- | PROD
   deriving (Ord, Eq)

data Flag    = Deadlock | Action | Config | Path | Prop
   deriving (Eq)

-- Some useful functions
     
dropCount :: Eq a => Int -> (a -> Bool) -> (a -> Bool) -> [a] -> (Int, [a])
dropCount acc cond until ls =
-- drops the shortest prefix of 'ls' whose elements do not satisfy
-- 'unil' while counting the elements that satisfy 'cond'
  case ls of
    [] -> (0, [])
    e:ls' ->
      if until e
      then (acc + (if (cond e) then 1 else 0), ls')
      else dropCount (acc + (if (cond e) then 1 else 0)) cond until ls'

intersect :: Ord a => Set a -> Set a -> Bool
intersect x y = not(S.null (S.intersection x y))

pairwiseDisjoint :: (Ord k, Ord a) => Map k (Set a) -> Maybe (k, k)
pairwiseDisjoint f =
-- returns the pair of keys (k1,k2) such that intersect (f!k1) (f!k2)
  case M.keys f of
    [] -> Nothing
    [_] -> Nothing
    x:y:_ ->
      if Misc.intersect (f!x) (f!y)
      then Just (x,y)
      else
        let f' = pairwiseDisjoint (M.delete y f)
        in
          case f' of
            Nothing -> pairwiseDisjoint (M.delete x f)
            _ -> f'

equivalenceClass :: (Eq a, Ord a) => Set (a,a) ->  a -> Set a
equivalenceClass rel z = S.fold (S.union) S.empty $ 
                         S.map (\(x,y) -> S.insert x (S.singleton y)) (S.filter (\(x,y) -> x == z || y ==z) rel)

cartProd :: (Ord a, Ord b) => Set a -> Set b -> Set (a,b)
cartProd sa sb = S.fromList $ [(x,y) | x <- (S.toList sa), y <- (S.toList sb)]

justFilter :: (Ord a) => Set (Maybe a) -> Set a
justFilter set = helper (S.toList set)
    where helper (x:xs) = case x of 
                            Just y -> S.insert y (helper xs)
                            Nothing -> helper xs
          helper [] = S.empty

dropElems :: Ord a => (a -> Bool) -> Set a -> Set a
dropElems f x = S.foldr (\e y -> if (f e) then y else (S.insert e y)) S.empty x

list2map :: [a] -> Map Int a
list2map l = M.fromList [(p,l!!p) | p <- [0 .. (L.length l) - 1 ] ]

inv :: (Ord a, Ord b) => Map a b -> Map b a
inv m = M.fromList $ (L.zip (M.elems m) (M.keys m))

getId :: (Eq a, Show a, Show k) => a -> [(k, a)] -> k
getId e m = case m of
              []        -> error ("Element " ++ show e ++ " not in the map " ++ show m)
              (i,e'):ms -> if e==e' then i else (getId e ms)

isSublist :: (Eq a) => [a] -> [a] -> Bool
isSublist l1 l2 =
  case l1 of
    []   -> True
    e:ls -> e € l2 && (isSublist ls l2)

range :: Int -> [Int]
range i
  | i < 0    = [i+1 .. 0]
  | otherwise = [0 .. i-1]

flatten :: [[a]] -> [a]
-- nomen omen
flatten ls =
  case ls of
    []    -> []
    l:lls -> [e | e <-l] ++ (flatten lls)

separators :: String
separators = "\n\t @.,;:()[]{}|+-*/^!?%"

isAlpha :: Char -> Bool
-- For parsing: strings are made of the non-separator characters
isAlpha c =
    not (c € separators)
  
isPtp :: String -> Bool
-- Names of participants have to begin with a capital letter
isPtp s =
  (s /= []) &&
  ( ((head s) € (['A'.. 'Z'])) &&
    (L.all isAlpha (tail s)) )

-- isNothing :: Ord a => Maybe a -> Bool
-- isNothing = \x -> x == Nothing

idxOffset :: (Num a, Eq b) => b -> [b] -> a -> a
-- PRE:  
-- POST: -1 if e not in l, otherwise first position of e in the list plus the offset o
idxOffset e l o
  | l == []   = -1
  | otherwise = if e == (head l) then o else (idxOffset e (tail l) (o+1))

rmChar :: Char -> String -> String
rmChar c (x:xs) = if x == c then rmChar c xs else x : (rmChar c xs)
rmChar _ [] = "" 

replaceChar :: Char -> Char -> String -> String
replaceChar c c' s = case s of
                      [] -> s
                      c'':s' -> (if c''==c then c' else c''):(replaceChar c c' s')

rmChars :: String -> String -> String
rmChars cs (x:xs) = if x € cs then rmChars cs xs else x : (rmChars cs xs)
rmChars _ [] = "" 

rmveStars :: String -> String            
rmveStars = \s -> rmChar '*' s 

rmQuotes :: String -> String            
rmQuotes = \s -> rmChar '\"' s

update :: Int -> a -> [a] -> [a]
update i v l
     | i == 0    = v:(tail l)
     | otherwise = (head l) : Misc.update (i-1) v (tail l)

(€) :: (Eq a) => a -> [a] -> Bool
x € y = L.elem x y
          
minR :: (Eq a) => [(a,a)] -> [(a,a)]
minR as = [ (n,m) | (n,m) <- as, L.all (\(_,t) -> n/=t) as ]

maxR :: (Eq a) => [(a,a)] -> [(a,a)]
maxR as = [ (n,m) | (n,m) <- as, L.all (\(t,_) -> m/=t) as ]

equivalenceRelation :: (Eq a, Ord a) => Set (a,a) -> Set (a,a)
-- Assume that (x,x) in ys for all x
equivalenceRelation ys = transitiveClosure (S.fromList $ addPairs $ S.toList ys)
    where addPairs ((x,y):xs) =  if x/=y
                                 then (x,y):((y,x):(addPairs xs))
                                 else (y,x):(addPairs xs)
          addPairs [] = []
        
          
transitiveClosure :: (Eq a, Ord a) =>  Set (a, a) -> Set (a, a)
-- adapted from http://stackoverflow.com/questions/19212558/transitive-closure-from-a-list-using-haskell
transitiveClosure closure
    | closure == closureUntilNow = closure
    | otherwise                  = transitiveClosure closureUntilNow
    where closureUntilNow =
            S.union closure (S.fromList
                              [(a, c) | (a, b) <- S.toList closure,
                               (b', c) <- S.toList closure, b == b'])

transitiveClosureParam :: (Eq a, Ord a) => (a -> a -> Bool) -> Set (a, a) -> Set (a, a)
transitiveClosureParam f closure 
  | closure == closureUntilNow = closure
  | otherwise                  = transitiveClosureParam f closureUntilNow
  where closureUntilNow = 
          S.union closure (S.fromList
                           [(a, c) | (a, b) <- S.toList closure,
                            (b', c) <- S.toList closure, ((f b b') || (f b' b))])

reflexoTransitiveClosure :: (Eq a, Ord a) =>  Set a -> Set (a, a) -> Set (a, a)
reflexoTransitiveClosure els closure =
-- PRE: closure must be a binary relation over the elements in els
-- POST: returns the reflexo-transitive closure of the relation in the 2nd argument
   S.union (transitiveClosure closure) (S.fromList [(a, a) | a <- S.toList els])

pairsof :: [a] -> [(a,a)]
pairsof xs = zip (everyOtherEl xs) (everyOtherEl (tail xs))
    where everyOtherEl xs' = (head xs') : (everyOtherEl (tail xs'))

--
-- Some IO utils
--

dirpath :: String
dirpath = "experiments/results/"

nop :: IO ()
nop = sequence_ []

myPrint :: Map String String -> Command -> String -> IO ()
myPrint flags cmd msg = if not(flags!"-v" == "") then putStrLn $ msgFormat cmd msg else return ()

myError :: Command -> String -> String
myError cmd msg = error $ msgFormat cmd msg

mkSep :: [String] -> String -> String
mkSep l sep =
  case l of
    [] -> ""
    s:[] -> s
    "": l' -> mkSep l' sep
    s:l' -> s ++ sep ++ (mkSep l' sep)

setFileNames :: String -> Map String String -> (String, String, String, String)
setFileNames f flags
  |"" == flags!"-o" = (dir, dir ++ baseFile, baseFile, takeExtension f)
  | True            = ("", "", flags!"-o", "")
  where baseFile = takeBaseName f
        dir      =
          let
            d = flags!"-d"
          in
            case d of
              "" -> if flags!"-o" == "" then
                      dirpath
                    else "."
              _  -> (if (d!!((length d) - 1) == pathSeparator) then d else d ++ [pathSeparator]) -- ++ baseFile ++ [pathSeparator]

rmExtension :: String -> String -> String
rmExtension ext s = if (L.drop i s == ext) then fst $ L.splitAt i s else s
  where i = L.length s - L.length ext

fst3 :: (a,b,c) -> a
fst3 (x,_,_) = x

snd3 :: (a,b,c) -> b
snd3 (_,x,_) = x

thr3 :: (a,b,c) -> c
thr3 (_,_,x) = x
  
writeToFile :: FilePath -> String -> IO()
writeToFile file content = writeFile file content

info :: Map Command [String]
-- associates a list of messages to each command; the first string of the list
-- must be a quick description of the command
info =
  M.fromList [
  (CG, ["Command line interface to ChorGram; run \'chorgram help\' for more info"]
  ),
  (GMC, ["for its generalised multiparty compatibility of a communicating system ",
         "[-c configfile] [-b bound] [-l] [-m multiplicity] [-sn] [-D detmode] [-d dirpath] [-ts] [-nf] [-cp cpattern] [-tp tpattern] [-v] filename",
         "default: configfile = ./aux/chorgram.config",
         "\t bound of buffers; default bound = 0",
         "\t mutiplicity = 0    (deprecated)",
         "\t -sn standard names of states ('q'<int>)",
         "\t -nf disable FIFO semantics",
         "\t dirpath = " ++ dirpath,
         "\t fontsize = 8",
         "\t cpattern = \"\"",
         "\t tpattern = \"- - - -\"",
         "\t detmode = no    possible values are: min | det | no"]
  ),
  (BGG, ["transform a petri net into a global graph",
        "[-d dirpath] [-v] filename",
        "default: dirpath = " ++ dirpath
        ]
  ),
  (GC, ["semantic of g-choreographies",
        "[-d dirpath] [-l] [--sloppy] filename [-rg]",
        "default: dirpath = " ++ dirpath,
        "\t --sloppy: suppresses the check for well-formedness; set by default",
        "\t -l: adds a legend; not set by default"
       ]
  ),
  (GC2DOT, ["return the dot format of a g-choreography",
            "[-v] [-d dirpath] [-fmt (gml | gc | gmldiff | sloppygml)] [-o destfile] filename",
            "\t-v diplays the path to the resulting file; not set by default",
            "default: dirpath = " ++ dirpath,
            "\t-fmt = gc"
           ]
  ),
  (GC2FSA, ["return the communicating system and the projections of g-choreography in the fsa format",
            "[-o pref] [-u iter] [-v] filename",
            "default: iter = -1",
            "\t pref is used to prefix the filenames where results are stored"]
  ),
  (PROJ, ["return projections of a g-choreography",
          "[-D (min | det | no)] [-fmt (dot | fsa)] [-u iter] [-v] filename [-p <blank-separated list of participants>]",
            "default: -D no",
            "\t -fmt fsa",
            "\t -v diplays the mapping index |--> ptp",
            "\t iter = -1 (non unfolding semantics)",
            "\t if -p option is not used, then all participants are projected; otherwise only those listed",
            "\t\t(-p option must be at the end of the command)"
         ]
  ),
  (UNFOLD, ["unfold a g-choreography",
           "n filename"]
  ),
  (PTPS, ["return the list of participants of a g-choreography",
          "ptps [-v] filename",
            "\t -v diplays the mapping index |--> ptp"
         ]
  ),
  (GC2POM, ["compute the pomset semantics of a g-choreography and saves it into a file",
            "gc2pom [-d dirpath] [-u iter] [-fmt (hs | gml)] [-v] filename",
            "default: dirpath = " ++ dirpath,
            "\t -fmt hs   generates the haskell representation, otherwise the graphml format",
            "\t iter = 1   this is the unfolding depth of loops"
           ]
  ),
  (TSOF, ["compute transition systems of a g-choreography",
            "gc2pom [-b bound] [-sn] [-nf] [-fmt (cms | fsa)] [-D (min | det | no)] [-l] [-v] filename",
            "default: -b bound of buffers; default bound = 0",
            "\t -sn standard names of states (q<int>); not set by default",
            "\t -nf disable FIFO semantics",
            "\t -fmt system in fsa or cms format; default -fmt fsa",
            "\t -D select minimisation, determinisation, or nothing; default -D no",
            "\t -l: adds a legend; not set by default"
           ]
  ),
  (POM2GC, ["compute the g-choreography of a pomset (if any)",
            "[-d dirpath] filename",
            "default: dirpath = " ++ dirpath
           ]
  ),
  (GC2GML, ["convert a g-choreography into the gml format unfoldling loops according to the -u option",
            "[-u iter] [-o output-file] filename",
            "default: iter = 1"
           ]
  ),
  (SYS, ["convert a system to various formats",
         "viewsystem [-fmt (hs | dot | fsa)] [-ptp n] [-v] [-sn] filename",
         "\t -sn standard names of states (q<int>); set by default",
         "\t -ptp n outputs only participant the n-th participant",
         "\t -fmt f output format is f (hs native haskell representation), fsa, or dot; default: fsa"
        ]
  ),
  (MIN, ["determinise or minimises CFSMs",
         "[-D (min | det | no)] [-d dirpath] [-v] [-o destfile] filename",
         "default: dirpath = " ++ dirpath,
         "\t -D min"
        ]
  ),
  (WB, ["check for well-branchedness",
         "[-v] filename"
       ]
  ),
  (WS, ["check for well-sequencedness",
         "[-v] filename"
       ]
  ),
  (WF, ["check for well-forkedness",
         "[-v] filename"
       ]
  ),
  (GC2LATEX, ["generate the latex of a g-choreography using some macros defined by eM",
           "[--vfactor f] [--hfactor f] [--vunit unit] [--hunit unit] filename",
            "default: --vfactor factor to scale vspace; default bound = .05",
            "\t --hfactor factor to scale hspace; default bound = 3",
            "\t --vunit latex unit for vertical space; default --vunit pt",
            "\t --hunit latex unit for horizontal space; default --hunit pt"
         ]
  ),
  (HGSEM, ["compute the hypergraph semantics of a g-choreography (deprecated)",
           "[-d dirpath] [-v] [--sloppy] filename",
           "default: dirpath = " ++ dirpath
          ]
  ),
  (CFSM2TIKZ, ["generate a textual description of a CFMS in tikz",
          "[-v] [--positioning (absolute | relative)] filename",
           "default: --positioning = absolute",
           "\t cfsm2tikz -v filename prints an help message and ignores the file"
         ]
  ),
  (TEST, ["generate a textual description of a CFMS in tikz",
          "[-v] [--positioning (absolute | relative)] filename",
           "default: --positioning = absolute"
         ]
  )
  -- (PROD, ["[-d dirpath] [-l] filename",
  --         "default: dirpath = " ++ dirpath
  --        ]
  -- )
  ]

usage :: Command -> String
usage cmd =
  let (descr, use) = (head $ info!cmd, tail $ info!cmd)
      msg = L.foldr (\s ln -> s ++ "\n\t" ++ ln) "" use
  in  cmdName cmd ++ ": " ++ descr ++ "\n   Usage: " ++ cmdName(cmd) ++ " " ++ msg

cmdName :: Command -> String
cmdName cmd =
  case cmd of
    BGG       -> "buildgc"
    CFSM2TIKZ -> "cfsm2tikz"
    CG        -> "chorgram"
    GC        -> "gc"
    GC2DOT    -> "gc2dot"
    GC2FSA    -> "gc2fsa"
    GC2GML    -> "gc2gml"
    GC2LATEX  -> "gc2latex"
    GC2POM    -> "gc2pom"
    TSOF      -> "tsof"
    GMC       -> "gmc"
    MIN       -> "minimise"
    POM2GC    -> "pom2gc"
    PROJ      -> "project"
    PTPS      -> "ptps"
    SYS       -> "viewsystem"
    UNFOLD    -> "gcunfold"
    WB        -> "wb"
    WS        -> "ws"
    WF        -> "wf"
    TEST -> "cfsm2tikz"
    HGSEM -> "hgsem"
--    PROD -> "cfsmprod"


msgFormat :: Command -> String -> String
msgFormat cmd msg =
  (cmdName cmd) ++ ":\t" ++ msg

defaultFlags :: Command -> Map String String
-- The default argument of each command
defaultFlags cmd =
  M.fromList (case cmd of
                 CG     -> [("-d",dirpath), ("-o", ""), ("-v","")]
                 GMC    -> [("-d",dirpath), ("-o", ""), ("-v",""),
                            ("-c", "./aux/chorgram.config"),
                            ("-m","0"),   -- multiplicity **deprecated**
                            ("-ts",""),
                            ("-b","0"),
                            ("-cp", ""),
                            ("-tp", "- - - -"),
                            ("-p", ""),
                            ("-D","no")    -- 'min' for minimisation, 'det' for determinisation, 'no' for nothing
                           ]
                 BGG    -> [("-d",dirpath), ("-o", ""), ("-v","")]
                 GC     -> [("-d",dirpath), ("-o", ""), ("-v",""),("-l",""), ("--sloppy","yes")]
                 GC2DOT -> [("-d",dirpath), ("-o", ""), ("-v",""), ("-fmt","gc")]
                 GC2FSA -> [("-d",dirpath), ("-o", ""), ("-v",""), ("-u", "-1")]
                 PROJ   -> [("-d",dirpath), ("-o", ""), ("-v",""), ("-D","no"), ("-u", "-1"), ("-fmt", "fsa")]
                 UNFOLD -> [("-d",dirpath), ("-o", ""), ("-v","")]
                 PTPS   -> [("-d",dirpath), ("-o", ""), ("-v","")]
                 GC2POM -> [("-d",dirpath), ("-o", ""), ("-v",""), ("-fmt", "hs"), ("-u","1")]
                 TSOF   -> [("-d",dirpath), ("-o", ""), ("-v",""), ("-fmt", "fsa"), ("-b", "0"), ("-D","no")]
                 POM2GC -> [("-d",dirpath), ("-o", ""), ("-v","")]
                 GC2GML -> [("-d",dirpath), ("-o", ""), ("-v",""), ("-u","1")] -- '-l' unfolding of loops
                 SYS    -> [("-d",dirpath), ("-o", ""), ("-v",""), ("-fmt", "fsa"), ("-sn", "yes")]
                 MIN    -> [("-d",dirpath), ("-o", ""), ("-v",""), ("-D","min")]
                 WB     -> []
                 WS     -> []
                 WF     -> []
                 GC2LATEX  -> [("--vfactor", "3"), ("--hfactor", "3"), ("--vunit", "pt"), ("--hunit", "pt")]
                 HGSEM  -> [("-d",dirpath), ("-o", ""), ("-v","")]
                 CFSM2TIKZ   -> [("--positioning", "absolute")]
                 TEST   -> [("--positioning", "absolute")]
                 --                      PROD     -> M.fromList [("-d",dirpath)]
              )

getCmd :: Command -> [String] -> (String, Map String String)
getCmd cmd args =
  let (al, fl) = L.splitAt ((length args) - 1) args
  in (head fl, getFlags cmd al)

getFlags :: Command -> [String] -> Map String String
getFlags cmd args =
  let yes = "yes" in
  case cmd of
    CG  ->  case args of
      []        -> defaultFlags(cmd)
      "-v":xs   -> M.insert "-v" yes (getFlags cmd xs)
      _         -> error $ usage(cmd)
    GMC -> case args of
      []                 -> defaultFlags(cmd)
      "-c":y:xs          -> M.insert "-c"  y      (getFlags cmd xs)
      "-l":xs            -> M.insert "-l" "no"    (getFlags cmd xs) -- turns off the leged of dot files
      "-rg":xs           -> M.insert "-rg" "rg"   (getFlags cmd xs) -- for reversible ggs TODO: add to manual
      "-ts":xs           -> M.insert "-ts" "ts"   (getFlags cmd xs)
      "-sn":xs           -> M.insert "-sn" yes    (getFlags cmd xs)      
      "-D":y:xs          -> M.insert "-D"  y      (getFlags cmd xs)
      "-b":y:xs          -> M.insert "-b"  y      (getFlags cmd xs)
      "-m":y:xs          -> M.insert "-m"  y      (getFlags cmd xs)
      "-d":y:xs          -> M.insert "-d"  y      (getFlags cmd xs)
      "-o":y:xs          -> M.insert "-o"  y      (getFlags cmd xs)
      "--fontsize":y:xs  -> M.insert "-fs" y      (getFlags cmd xs)
      "-nf":xs           -> M.insert "-nf" "no"   (getFlags cmd xs) -- turns off the fifo policy
      "-cp":y:xs         -> M.insert "-cp" y      (getFlags cmd xs)
      "-tp":y:xs         -> M.insert "-tp" y      (getFlags cmd xs)
      "-p":y:xs          -> M.insert "-p"  y      (getFlags cmd xs)
      "-v":xs            -> M.insert "-v"  yes    (getFlags cmd xs) -- turn on verbose mode
      _                  -> error $ usage(cmd)
    BGG  ->  case args of
      []        -> defaultFlags(cmd)
      "-d":y:xs -> M.insert "-d" y   (getFlags cmd xs)
      "-v":xs   -> M.insert "-v" yes (getFlags cmd xs)
      _         -> error $ usage(cmd)
    GC -> case args of
      []            -> defaultFlags(cmd)
      "-d":y:xs     -> M.insert "-d"  y        (getFlags cmd xs)
      "-rg":xs      -> M.insert "-rg" yes      (getFlags cmd xs)
      "-v":xs       -> M.insert "-v" yes       (getFlags cmd xs)
      "-l":xs       -> M.insert "-l" yes       (getFlags cmd xs)
      "--sloppy":xs -> M.insert "--sloppy" yes (getFlags cmd xs)
      _             -> error $ usage(cmd)
    GC2DOT -> case args of
      []            -> defaultFlags(cmd)
      "-fmt":y:xs  -> M.insert "-fmt" y (getFlags cmd xs)
      "-d":y:xs     -> M.insert "-d" y    (getFlags cmd xs)
      "-v":xs       -> M.insert "-v" yes  (getFlags cmd xs)
      "-o":y:xs     -> M.insert "-o" y    (getFlags cmd xs)
      _             -> error $ usage(cmd)
    GC2FSA -> case args of
      []            -> defaultFlags(cmd)
      "-v":xs       -> M.insert "-v" yes (getFlags cmd xs)
      "-u":y:xs     -> M.insert "-u" y (getFlags cmd xs)
      "-o":y:xs     -> M.insert "-o" y   (getFlags cmd xs)
      _             -> error $ usage(cmd)
    GC2GML -> case args of
      []            -> defaultFlags(cmd)
      "-v":xs       -> M.insert "-v" yes (getFlags cmd xs)
      "-l":y:xs     -> M.insert "-l" y   (getFlags cmd xs)
      "-d":y:xs     -> M.insert "-d" y   (getFlags cmd xs)
      "-o":y:xs     -> M.insert "-o" y   (getFlags cmd xs)
      _             -> error $ usage(cmd)
    PROJ -> case args of
      []            -> defaultFlags(cmd)
      "-v":xs       -> M.insert "-v" yes (getFlags cmd xs)
      "-fmt":y:xs  -> M.insert "-fmt" y (getFlags cmd xs)
      "-D":y:xs     -> M.insert "-D" y (getFlags cmd xs)
      "-u":y:xs     -> M.insert "-u" y (getFlags cmd xs)
      _             -> error $ usage(cmd)
    UNFOLD -> case args of
      []            -> defaultFlags(cmd)
      "-v":xs       -> M.insert "-v" yes (getFlags cmd xs)
      _             -> error $ usage(cmd)
    PTPS -> case args of
      []            -> defaultFlags(cmd)
      "-v":xs       -> M.insert "-v" yes (getFlags cmd xs)
      _             -> error $ usage(cmd)
    SYS -> case args of
      []           -> defaultFlags(cmd)
      "-fmt":y:xs -> M.insert "-fmt" y (getFlags cmd xs)
      "-sn":xs     -> M.insert "-sn" yes (getFlags cmd xs)
      "-ptp":y:xs  -> M.insert "-ptp" y (getFlags cmd xs)
      "-v":xs      -> M.insert "-v" yes (getFlags cmd xs)
      _            -> error $ usage(cmd)
    MIN -> case args of
      []        -> defaultFlags(cmd)
      "-d":y:xs -> M.insert "-d" y   (getFlags cmd xs)
      "-v":xs   -> M.insert "-v" yes (getFlags cmd xs)
      "-D":y:xs -> M.insert "-D" y   (getFlags cmd xs)
      "-o":y:xs -> M.insert "-o" y   (getFlags cmd xs)
      _         -> error $ usage(cmd)
    WB -> case args of
      []        -> defaultFlags(cmd)
      "-v":xs   -> M.insert "-v" yes (getFlags cmd xs)
      _         -> error $ usage(cmd)
    WS -> case args of
      []        -> defaultFlags(cmd)
      "-v":xs   -> M.insert "-v" yes (getFlags cmd xs)
      _         -> error $ usage(cmd)
    WF -> case args of
      []        -> defaultFlags(cmd)
      "-v":xs   -> M.insert "-v" yes (getFlags cmd xs)
      _         -> error $ usage(cmd)
    GC2LATEX -> case args of
      []            -> defaultFlags(cmd)
      "--vfactor":y:xs   -> M.insert "--vfactor" y (getFlags cmd xs)
      "--hfactor":y:xs   -> M.insert "--vfactor" y (getFlags cmd xs)
      "--vunit":y:xs -> M.insert "--vunit" y (getFlags cmd xs)
      "--hunit":y:xs -> M.insert "--hunit" y (getFlags cmd xs)
      _         -> error $ usage(cmd) ++ "\tbad pattern"
    HGSEM -> case args of
      []            -> defaultFlags(cmd)
      "--sloppy":xs -> M.insert "--sloppy" yes (getFlags cmd xs)
      "-v":xs   -> M.insert "-v" yes (getFlags cmd xs)
      "-d":y:xs -> M.insert "-d"  y        (getFlags cmd xs)
      _         -> error $ usage(cmd) ++ "\tbad pattern"
    -- PROD -> case args of
    --   []        -> defaultFlags(cmd)
    --   "-d":y:xs -> M.insert "-d"  y    (getFlags cmd xs)
    --   _         -> error $ usage(cmd)
    GC2POM -> case args of
      []           -> defaultFlags(cmd)
      "-fmt":y:xs -> M.insert "-fmt" y (getFlags cmd xs)
      "-d":y:xs    -> M.insert "-d" y    (getFlags cmd xs)
      "-u":y:xs    -> M.insert "-u" y    (getFlags cmd xs)
      "-v":xs      -> M.insert "-v" yes  (getFlags cmd xs)
      _            -> error $ usage(cmd)
    TSOF -> case args of
      []           -> defaultFlags(cmd)
      "-fmt":y:xs -> M.insert "-fmt" y  (getFlags cmd xs)
      "-D":y:xs    -> M.insert "-D" y     (getFlags cmd xs)
      "-sn":xs     -> M.insert "-sn" yes  (getFlags cmd xs)
      "-nf":xs     -> M.insert "-nf" "no" (getFlags cmd xs) -- turns off the fifo policy
      "-b":y:xs    -> M.insert "-b" y     (getFlags cmd xs)
      "-l":xs      -> M.insert "-l" yes   (getFlags cmd xs)
      "-v":xs      -> M.insert "-v" yes   (getFlags cmd xs)
      _            -> error $ usage(cmd)
    POM2GC -> case args of
      []        -> defaultFlags(cmd)
      "-d":y:xs -> M.insert "-d"  y        (getFlags cmd xs)
      _         -> error $ usage(cmd)
    CFSM2TIKZ -> case args of    -- testing GC2LATEX
      []            -> defaultFlags(cmd)
      "--positioning":y:xs -> M.insert "--positioning" y (getFlags cmd xs)
      "-v":xs -> M.insert "-v" yes (getFlags cmd xs)
      _         -> error $ usage(cmd)
    TEST -> case args of
      []            -> defaultFlags(cmd)
      "--positioning":y:xs -> M.insert "--positioning" y (getFlags cmd xs)
      "-v":xs -> M.insert "-v" yes (getFlags cmd xs)
      _         -> error $ usage(cmd)
--
-- Some utilities on graphs
--

fanOut :: Ord vertex => Ord label => Graph vertex label -> vertex -> Set (Edge vertex label)
fanOut ( _, _, _, trxs ) q = S.filter (\(x,_,_) -> x == q) trxs

pClosure :: Ord vertex => Ord label => Graph vertex label -> (label -> Bool) -> vertex -> Set vertex
pClosure g lpred v =
--  PRE:
--  POST: returns the closure of vertexes reachable from v with
--        transitions that satisfy the predicate lpred on labels
  let aux res wl visited =
        case wl of
          []     -> S.insert v res
          v':wl' -> if v' € visited
                    then aux res wl' visited
                    else aux vs' wl'' (v':visited)
            where vs'  = S.foldl S.union res
                         (S.map (\(q,l,q') -> if (lpred l) && q == v' then (S.singleton q') else S.empty) (edgesOf g))
                  wl'' = wl' ++ S.toList vs'
  in aux S.empty [v] []

replaceVertex :: Ord vertex => Ord label => vertex -> vertex -> Graph vertex label -> Graph vertex label
replaceVertex q q' g@(vs, q0, labels, trxs) =
-- replace q with q' in g (substitution q |--> q')
  if q == q'
  then g
  else (S.insert q' (S.delete q vs), (aux q0), labels, trxs')
    where trxs'  = S.map (\( q_, l, q_' ) -> ( aux q_, l, aux q_' )) trxs
          aux q_ = if q_ == q then q' else q_

  
replaceVertexes :: Ord vertex => Ord label => (vertex -> Bool) -> vertex -> Graph vertex label -> Graph vertex label
replaceVertexes cond q g@( vs, _, _, _ ) =
-- replace each vertex satisfying cond with q in g
  let
    aux g' q' =
      if cond q'
      then replaceVertex q q' g'
      else g'
  in
    S.foldl aux g (vs)


pRemoval :: Ord vertex => Ord label => Graph vertex label -> (label -> Bool) -> Graph vertex label
-- Generalisation of the epsilon-removal in NFA to p-removal for a
-- predicate lpred on labels
pRemoval g@(vs, v0, _, trxs) lpred = (vs, v0, S.map glabel trxs', trxs')
  where
    (_, other_trxs) = S.partition (lpred . glabel) trxs
    aux (s, l, s') =
      case lpred l of
        True  ->
          S.fromList $
          L.map (\(l', q) -> (s, l', q)) [(glabel t, gtarget t) | t <- (S.toList other_trxs), gsource t € (S.toList $ pClosure g lpred s')]
          ++
          L.map (\(q, l') -> (q, l', s')) [(gsource t, glabel t) | t <- (S.toList other_trxs), s € (S.toList $ pClosure g lpred (gtarget t))]
        False -> S.map (\q -> (s, l, q)) (S.delete s' (pClosure g lpred s'))
    new_trxs = S.unions $ S.toList $ S.map aux trxs
    trxs' = S.union other_trxs new_trxs


eqClassOf :: (Ord a) => a -> [Set a] -> Set a
-- returns the first set in 'classes' containing 'state'
eqClassOf state classes =
  case classes of
    []          -> S.empty
    qs:classes' -> if (S.member state qs) then qs else (eqClassOf state classes')

reachableVertexes :: Ord vertex => Ord label => Graph vertex label -> vertex -> Set vertex
reachableVertexes g =
  --  PRE:
  --  POST: returns the set of vertexes of g reachable from a given vertex
  pClosure g (\_ -> True)

states2int :: Ord vertex => Ord label => Graph (Set vertex) label -> Graph Int label
states2int (qs, q0, ls, trxs) =
  let
    dict = M.fromList [(q, S.findIndex q qs) | q <- S.toList qs]
  in
    (S.fromList $ M.elems dict, dict!q0, ls, S.map (\(q,l,q') -> (dict!q, l, dict!q')) trxs)

-- Projections of Graph and Edge components

gnodes :: Graph vertex label -> Set vertex
gnodes (vertexes, _, _, _) = vertexes

ginitialnode :: Graph vertex label -> vertex
ginitialnode (_, v, _, _) = v

initialnode :: Graph vertex label -> vertex
initialnode (_, v, _, _) = v

edgesOf :: Graph vertex label -> Set (Edge vertex label)
edgesOf (_, _, _, trans) = trans

gsource :: Edge vertex label -> vertex
gsource (v, _, _) = v

glabel :: Edge vertex label -> label
glabel (_, l, _) = l

gtarget :: Edge vertex label -> vertex
gtarget (_, _, v) = v

-- Some utilities

notNothing :: Eq a => Maybe a -> Bool
notNothing = \x -> x /= Nothing

fstJust :: Eq a => [Maybe a] -> Maybe a
fstJust l =
  case l of
    [] -> Nothing
    Nothing : l' -> fstJust l'
    x : _ -> x

sn states =
  -- standard renaming of CFSMs' states
  M.fromList [((S.toList states)!!i, "q" ++ show i)
             | i <- range (S.size states)]

grenameVertex :: Ord vertex => Ord label => Map vertex vertex -> Graph vertex label -> Graph vertex label
grenameVertex sigma (nodes, n0, labels, trans) = (nodes', n0', labels, trans')
  where nodes' = S.map aux nodes
        n0'    = aux n0
        trans' = S.map (\(n, e, n') -> (aux n, e, aux n')) trans
        aux n  = if M.member n sigma then sigma!n else n

isTerminal ::  Eq vertex => vertex -> Graph vertex label -> Bool
isTerminal q (_,_,_,trxs) =
  let l = S.toList trxs in q € [q_ | (_,_,q_) <- l, L.all (\(q',_,_) -> q' /= q) l]

goutgoing :: Eq vertex => Graph vertex label -> vertex -> Set (Edge vertex label)
-- goutgoing gr v
--  PRE:  v is a vertex of gr
--  POST: returns the outgoing edges of v in gr
goutgoing (_, _, _,trans) v = S.filter (\t -> v == (gsource t)) trans

gincoming :: Eq vertex => Ord vertex => Graph vertex label -> Map vertex (Set (Edge vertex label))
-- gincoming gr
--  PRE:  
--  POST: builds a map associating to each vertex of gr its incoming edges
gincoming (vertexes, _, _,trans) = M.fromList $ i
    where i = L.map (\v -> (v, S.filter (\t -> v == (gtarget t)) trans)) (S.toList vertexes)


gpath :: Eq vertex => (Show label) => (Show vertex) => Graph vertex label -> vertex -> vertex -> [vertex] -> [[Edge vertex label]]
-- gpath gr s t l
--   PRE:  s, t veterexes in gr, l list of veterexes of gr
--   POST: returns the list of (elementary) paths in gr from s to t not passing trough veterexes in l
gpath gr s t l
    | s == t    = if (s € l)
                  then []
                  else ([[]] ++ [tr:p | tr@(s0,_,s') <- s_outgoing, s0 /= s', p <- gpath gr s' t l])
    | otherwise = [tr:p | tr@(_, _, s') <- s_outgoing, p <- (paths s')]
    where s_outgoing = [(s0, e, s') | (s0, e, s') <- (S.toList $ goutgoing gr s), not(s' € l)]
          paths v = gpath gr v t ([s] ++ l)

gpath_ :: Eq vertex => Ord vertex => Eq label => (Show label) => (Show vertex) =>
          Graph vertex label ->
              Map vertex (Set (Edge vertex label)) ->
                  Map vertex [[Edge vertex label]] ->
                      vertex ->
                          vertex ->
                              [Edge vertex label] ->
                                  Map vertex [[Edge vertex label]]
-- cp current paths
gpath_ gr incoming cp s t l
    | s == t    = cp
    | otherwise = addtr (S.toList $ incoming!t) l cp
    where addpath _ [] m     = m
          addpath c (p:ps) m = M.insert c (if c € M.keys m' then (cp!c ++ [p]) else [p]) m'
              where m' = addpath c ps m
          paths arc@(u, _, _) visited current = if arc € visited
                                                then current
                                                else gpath_ gr incoming current s u (arc:visited)
          addtr [] _ cp' = cp'
          addtr (tr@(v,_,_):trs) l' cp'
              | tr € l'    = addtr trs l' cp'
              | otherwise =  cp''
              where tov = paths tr l' cp'
                    tot = if M.null tov || not(v € M.keys tov) || [[]] == tov!v
                             then if v == s then [[tr]] else [[]]
                           else L.map (\p -> p ++ [tr]) (if v € M.keys tov then tov!v else [[]])
                    cp'' = addtr trs (tr:l') (addpath t tot tov)

gpath' :: Eq vertex => Ord vertex => Eq label => (Show label) => (Show vertex) =>
          Graph vertex label ->
              Map vertex (Set (Edge vertex label)) ->
                  Map vertex [[Edge vertex label]] ->
                      vertex ->
                          [vertex] ->
                              [Edge vertex label] ->
                                  Map vertex [[Edge vertex label]]
gpath' gr incoming cp s trs l
    | L.null trs = cp
    | s == t     = gpath' gr incoming cp s (tail trs) l
    | L.null itr = gpath' gr incoming cp s (tail trs) l
    | otherwise  = gpath' gr incoming cp' s (v:trs) (tr:l)
    where t = head trs
          itr = (S.toList (incoming!t)) L.\\ l
          tr@(v, _, _)  = head itr
          cp' = case (v € M.keys cp) of
                  False -> addpath t [[tr]] cp
                  True  -> if [[]] == cp!v || L.null(cp!v)
                             then addpath t [[tr]] cp
                           else addpath t [p ++ [tr] | p <- cp!v] cp
          addpath _ [] m     = m
          addpath c (p:ps) m = M.insert c (if c € M.keys m' then (m'!c ++ [p]) else [p]) m'
              where m' = addpath c ps m


underscore :: String
underscore = "UNDERSCORE"

plus :: String
plus = "PLUS"

minus :: String
minus = "MINUS"

times :: String
times = "TIMES"

eq :: String
eq = "EQ"

fwdslash :: String
fwdslash = "FWDSLASH"

bckslash :: String
bckslash = "BCKSLASH"

orbra :: String
orbra = "ORBRA"

crbra :: String
crbra = "CRBRA"

osbra :: String
osbra = "OSBRA"

csbra :: String
csbra = "CSBRA"

ocbra :: String
ocbra = "OCBRA"

ccbra :: String
ccbra = "CCBRA"

eee :: String
eee = "EEE"

acute :: String
acute = "ACUTE"

precent :: String
precent = "PRECENT"

dollar :: String
dollar = "DOLLAR"

sharp :: String
sharp = "SHARP"

at :: String
at = "AT"

emark :: String
emark = "EMARK"

qmark :: String
qmark = "QMARK"

plusminus :: String
plusminus = "PLUSMINUS"

para :: String
para = "PARA"

accent :: String
accent = "ACCENT"

colon :: String
colon = "COLON"

semicolon :: String
semicolon = "SEMICOLON"

par :: String
par = "PAR"

lt :: String
lt = "LT"

gt :: String
gt = "GT"

comma :: String
comma = "COMMA"

period :: String
period = "PERIOD"

quote :: String
quote = "QUOTE"

tilde :: String
tilde = "TILDE"

blank :: String
blank = "BLANK"

tokenifychar :: Char -> String
tokenifychar c = case c of
  '_'  -> underscore
  '+'  -> plus
  '-'  -> minus
  '*'  -> times
  '='  -> eq
  '/'  -> fwdslash
  '\\' -> bckslash
  '('  -> orbra
  ')'  -> crbra
  '['  -> osbra
  ']'  -> csbra
  '{'  -> ocbra
  '}'  -> ccbra
  '&'  -> eee
  '^'  -> acute
  '%'  -> precent
  '$'  -> dollar
  '#'  -> sharp
  '@'  -> at
  '!'  -> emark
  '?'  -> qmark
  '±'  -> plusminus
  '§'  -> para
  '\'' -> accent
  ':'  -> colon
  ';'  -> semicolon
  '|'  -> par
  '<'  -> lt
  '>'  -> gt
  ','  -> comma
  '.'  -> period
  '\"' -> quote
  '~'  -> tilde
  ' '  -> blank
  _    -> [c]

-- change to String -> String to replace substrings with the right symbol
-- detokenifychar :: String -> String

detokenifyTable :: [(String,String)]
detokenifyTable = [
  (underscore ,  "_"),
  (plus , "+"),
  (minus , "-"),
  (times , "*"),
  (eq , "="),
  (fwdslash , "/"),
  (bckslash , "\\"),
  (orbra , "("),
  (crbra , ")" ),
  (osbra , "["),
  (csbra , "]"),
  (ocbra , "{"),
  (ccbra , "}"),
  (eee , "&" ),
  (acute , "^"),
  (precent , "%"),
  (dollar , "$"),
  (sharp , "#"),
  (at , "@"),
  (emark , "!"),
  (qmark , "?"),
  (plusminus , "±"),
  (para , "§"),
  (accent , "\""),
  (colon , ":"),
  (semicolon , ";"),
  (par , "|"),
  (lt , "<"),
  (gt , ">"),
  (comma , ","),
  (period , "."),
  (quote , "\""),
  (tilde , "~"),
  (blank , " ")
  ]

tokenifymsg :: Message -> String
tokenifymsg msg = L.concat $ L.map tokenifychar msg

debug :: (Show a) => a -> b
debug x = error (show $ x)

mkFileName :: String -> String -> String -> String -> String
mkFileName suf dir fname ext =
  dir ++ fname ++ (rmChar '\"' suf) ++ ext

-- Some strings and an auxiliary function to handle node creation
lpref :: String
epref :: String
(lpref, epref) = ( "*<" , ">*" )

newNode :: (Int -> Bool) -> Int -> Int -> Int
newNode excluded n j = if (excluded n) then n else (if n > 0 then n + j else n - j)

type DotString = String

getConf :: String -> IO(Map String DotString)
getConf f = do
  conf <- readFile f
  let
    aux   =
      \l ->
        L.map (\p ->
                 (T.unpack $ p!!0, T.unpack $ p!!1)
              ) [T.words l |
                 ((L.length $ T.unpack l) > 2) &&
                 (L.take 2 (T.unpack l) /= "--")
                           ]
  let lns
        = T.lines $ T.pack conf
  return (M.fromList $ L.concat $ L.map aux lns)

tabsize = 3::Int
