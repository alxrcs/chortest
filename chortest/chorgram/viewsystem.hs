--
-- Authors: Emilio Tuosto <emilio.tuosto@gssi.it>
--
-- Transforming a communicating system and in various formats among
-- which the corresponding Haskell's data structure
--

import Misc
import CFSM
import DotStuff
import SystemParser
import Data.List as L
import Data.Map.Strict as M
import System.Environment

main :: IO ()
main = do progargs <- getArgs
          if L.null progargs
            then putStr $ usage SYS
            else do
              let (sourcefile, flags) =
                    getCmd SYS progargs
              myPrint flags SYS ("parsing started" ++ "\n" ++ (show flags))
              txt <- readFile sourcefile
              let (_, _, _, ext) =
                    setFileNames sourcefile flags
              flinesIO <- getDotConf
              let flines =
                    M.fromList [(key, flinesIO!key) | key <- M.keys flinesIO]
              let (sys, ptps) = parseSystem ext txt
              let system = 
                    (if (M.member "-sn" flags)
                     then L.map (\cfsm -> (grenameVertex (sn $ statesOf cfsm) cfsm)) sys
                     else sys,
                     ptps
                    )
              let toprint =
                    if (M.member "-ptp" flags)
                    then read (flags!"-ptp")::Int
                    else (M.size ptps) + 1
              let res =
                    case flags!"-fmt" of
                      "fsa" ->
                        if toprint > M.size ptps
                        then system2String system 
                        else cfsm2String (ptps!toprint) (sys!!toprint)
                      "hs" ->
                        if toprint > M.size ptps
                        then show system
                        else show (sys!!toprint)
                      "dot" ->
                        if toprint > M.size ptps
                        then dottifySystem flines system
                        else dottifyCFSM (sys!!toprint) (ptps!toprint) "" flines
                      _ -> "Unknown format " ++ (flags!"-fmt")
              putStrLn res
