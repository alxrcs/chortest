--
-- Authors: Emilio Tuosto <emilio.tuosto@gssi.it>
--
-- This program unfolds a g-choreography.
-- 

import Misc
import GCParser
import SyntacticGlobalChoreographies (unfoldGC, gc2txt)
import System.Environment
import Data.List as L

main :: IO ()
main = do
  progargs <- getArgs
  if L.length progargs /= 2
    then do putStrLn $ usage UNFOLD
    else do
      let n = read (head progargs) :: Int
          sourcefile = head $ tail progargs
      gctxt <- readFile sourcefile
      let ( gc, _ ) =
            case gcgrammar gctxt (0, 0) (0, 0) of
              Ok x -> x
              Er err -> error err
          gc' = unfoldGC n gc
      putStrLn (gc2txt 0 gc')
