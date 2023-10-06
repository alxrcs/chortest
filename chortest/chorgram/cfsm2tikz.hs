--
-- Authors: Emilio Tuosto <emilio.tuosto@gssi.it>
--

import Misc
import SystemtoTikz
import Data.List as L (null, intercalate, map, length, zip, sort, foldr)
import System.Environment (getArgs)
import Data.Map.Strict as M
  
main :: IO ()
main = do
  progargs <- getArgs
  if L.null progargs
    then putStrLn $ usage CFSM2TIKZ
    else do
    let ( sourcefile, flags ) = getCmd CFSM2TIKZ progargs
    let mknode =
          if flags!"--positioning" == "absolute"
          then mkabsolute
          else mkrelative
    let aux = \k t -> k ++ " " ++ (flags!k) ++ " " ++ t
    let comment = "% cfsm2tikz " ++
          (L.foldr aux "" (M.keys flags)) ++
          sourcefile ++ "\n% "
    if (M.member "-v" flags)
      then do putStrLn $ SystemtoTikz.verbose sourcefile
      else do
        (if sourcefile == "" then return "" else readFile sourcefile) >>=
          \txt ->
            case cfsmtotikz txt (1,0) of
              Ok tikz ->
                putStrLn $ comment ++ (tikz mknode)
              Er err -> putStrLn err
