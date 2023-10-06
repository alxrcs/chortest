Description of the case study
-----------------------------

The EU Customs Business Process Models were created in 2010, by request of the Member States' customs authorities, to facilitate the reading of the proposed legal provisions. Their purpose is to show the legal provisions in a visual, understandable and integrated way. These are composed by the most part of BPM Notation (BPMN) diagrams and Value Added Chain Diagrams (VACD), publicly available online at [0].

The "Entry of Goods" comprises multiple processes which depend on multiple factors, such as where the Entry Summary Declaration (ENS) document was lodged, and the medium through which the goods enter (i.e. Sea, Air, or Road and Rail). Amendments to said ENS can also be received, and additionally, involved parties can be notified if requested.

To illustrate one of these processes, let us briefly describe how an Amendment gets processed on the Customs Office Of First Entry (COFE) [1]. Upon being received by the COFE, it goes through an initial verification process, which checks, for example, whether the person submitting the amendment attempts to amend only particulars that this person had previously submitted. 
If the ENS Amendment is deemed valid, it is registered and multiple participants are notified. If the ENS Amendment was lodged at the COFE, a registration notification is sent to the Declarant or Representative (these terms will be used interchangeably from now one), and the Carrier is also optionally notified (when different from the Declarant and requests to be notified). If the Amendment was lodged somewhere else instead, a notification of the registration is sent to the Other Customs Office than COFE. If the ENS Amendment is invalid instead, the participants are notified of the rejection following a similar communication pattern as before. 

A more elaborate process is the arrival of new goods through the aforementioned mediums. For the purposes of illustration, let us focus on the case where the goods arrive through "Road and Rail" [2](). When a Full or Partial ENS is received at the Declared Customs of First Entry, a first check occurs for the validity and consistency of the information provided. If the ENS is deemed invalid, it is rejected, and this is notified to the Declarant. Otherwise, it is registered and is automatically assigned a Master Reference Number (MRN), which is stored in the system. This registration is notified to the Declarant, as well as the Carrier (when different from the Declarant and it requested to be notified). Up to this point, the Declarant can optionally send Amendments, which are processed by the COFE as mentioned before.
At this point, a risk analysis needs to occur, and the ENS is forwarded to the member states to be involved in the risk analysis, and the custom offices that are deemed to be involved. These offices should send back information about any identified risks.
If any additional information is required by the COFE and the ENS is lodged at the COFE, the Declarant and (optionally) the Carrier are notified, and the COFE awaits for further information from the Declarant. If the ENS was not lodged at the COFE, the notification for more information is sent to the corresponding Customs Office other than the COFE (OCOFE).
There might be the need for additional controls to be performed, in which case some of the involved actors might need to be notified, in a similar fashion as when additional information is required (i.e. the Declarant, the Carrier and the OCOFE).
Once the additional controls (if any) are performed, the ENS Data Risk results and control measures should be forwarded to the involved member states customs offices.

(i) assumptions forced by the constraints imposed
by g-choreographies 
-------------------------------------------------
* If well-formedness is desired, some of the choices will not be well branched 
due to the fact that the choice to be notified is not made explicit in the 
protocol by the participant who should make it. Examples of this are whether the 
carrier should be notified when more information is required (or in case of rejection)
in the Road-Rail process.
* There is a time limit to perform the risk analysis which is not modeled 
in the associated g-choreography.
* Under some circumstances, Amendments to the ENS can be received while the ENS itself is being processed, as illustrated in [2].
Due to the nature of BPMN diagrams, the reception of these messages also sends tokens to specific points in the protocol's execution, which makes it considerably
harder (and in some cases outright disallow) to model it using structured g-choreographies.


(ii) the assumptions made to just to simplify the
model (not due to restricted expressiveness), and 
-------------------------------------------------
* To reduce the amount of branching and avoid repetition, the ENS is assumed to be
always lodged at the COFE.


(iii) some examples of
assumptions/guesses made due to lack of clarity 
in the official documentation.
-------------------------------------------------
* It is unclear from the protocol description, whether the Amendments are sent in parallel or as a choice by specific participants at a specific point in the protocol.
Due to this consideration, they were left out of the model for the Road-Rail entry case (NOTE: this is the same as the last assumption in cat.(i)).


References
----------

[0](https://aris9.itsmtaxud.eu/businesspublisher/login.do?login=anonymous&password=anonymous)
[1](L3-ENT-01-05-Process ENS At Office Of First Entry - Amendment)
[2](L3-ENT-01-04-Process ENS At Office Of First Entry - Road and Rail)