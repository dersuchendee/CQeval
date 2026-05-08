"""
Apply scenario corrections to benchmarkdataset.csv
"""
import pandas as pd

CSV_PATH = str(BASE_DIR / "data" / "benchmarkdataset_polifonia.csv")

df = pd.read_csv(CSV_PATH)

REPLACEMENTS = {}

# ── 1. Amy organ builders ─────────────────────────────────────────────────────
REPLACEMENTS[
    'Amy wants to assess the developments of organ builders. This research includes looking into which organs an organ builder worked on and how their projects developed over time. This development can range from an increase of the size of the organ, to locality of their projects, or to the type or prestige of the project. The history section of the 15-part organ encyclopaedia is mostly used for this as this specifies year and changes that are made to the organs, but this obviously takes a lot of time and effort to research.'
] = (
    'Amy wants to assess the developments of organ builders. This research includes looking into which organs an organ builder worked on and how their projects developed over time. This development can range from an increase of the size of the organ, to locality of their projects, or to the type or prestige of the project. The history section of the 15-part organ encyclopaedia is mostly used for this as this specifies year and changes that are made to the organs, but this obviously takes a lot of time and effort to research. \n'
    '    Amy already has access to the physical as well as online library of the university where she works.\n'
    '    Amy also has the entire 15 volume collection from the Dutch organ encyclopaedia. She uses the encyclopaedia mostly for reading about the arthistorical and technical aspects and developments of organs and comparing these to other organs.\n'
    '    Database websites for information about organs and churches which all offer slightly different kinds of information:\n'
    '        https://www.kerk-en-orgel.nl\n'
    '        https://reliwiki.nl/\n'
    '        http://www.orgbase.nl\n'
    '        https://geoplaza.vu.nl/cms/research/kerkenkaart/ \u2013 Kerkenkaart (churchmap) from the Vrije Universiteit Amsterdam\n'
)

# ── 2. Anna haptic concert ────────────────────────────────────────────────────
REPLACEMENTS[
    "Anna, who became hearing impaired in later life, attends a concert by The Blockheads at the Stables music venue with a friend Ben who is deaf from birth, and her hearing friend Caroline. In order to enhance their experience of the concert, they decide to use the haptic bracelets offered by the venue. Anna is a big fan of the bass player Norman Watt Roy, whereas her friend is interested in the drums played by John Roberts. For tonight's performance, the Stables are offering a live automatic haptic transcription of the bass part. The Stables have also hired a musical interpreter for the hearing impaired who will be communicating the overall performance using an electronic drum kit from which a haptic feed is available. As a third option, a chest-worn transducer is available that takes the audio feed and turns the wearer's chest cavity into a bass woofer. Anna used the bass transcription. Ben opted for the live interpretation. During 'Hit me with your rhythm stick' Anna grabs the hand of her hearing friend Caroline in excitement and puts it on her wrist to feel the bass rhythm. Anna and Ben sign to each other during the first two songs, and decide to swap feeds for the third.\n\nWhen they hand the equipment back in at the end of the gig, the Stables staff point out that there is a summer workshop being run for hearing impared beginner drummers using a full drum kit and bracelets on all four limbs."
] = (
    "Anna wants to experience live music with the aid of haptic technology. Anna, who became hearing impaired in later life, attends a concert by The Blockheads at the Stables music venue with a friend Ben who is deaf from birth, and her hearing friend Caroline. In order to enhance their experience of the concert, they decide to use the haptic bracelets offered by the venue. Anna is a big fan of the bass player Norman Watt Roy, whereas her friend is interested in the drums played by John Roberts. For tonight's performance, the Stables are offering a live automatic haptic transcription of the bass part. The Stables have also hired a musical interpreter for the hearing impaired who will be communicating the overall performance using an electronic drum kit from which a haptic feed is available. As a third option, a chest-worn transducer is available that takes the audio feed and turns the wearer's chest cavity into a bass woofer. Anna used the bass transcription. Ben opted for the live interpretation. During 'Hit me with your rhythm stick' Anna grabs the hand of her hearing friend Caroline in excitement and puts it on her wrist to feel the bass rhythm. Anna and Ben sign to each other during the first two songs, and decide to swap feeds for the third.\n\nWhen they hand the equipment back in at the end of the gig, the Stables staff point out that there is a summer workshop being run for hearing impared beginner drummers using a full drum kit and bracelets on all four limbs."
)

# ── 3. David brass bands ──────────────────────────────────────────────────────
REPLACEMENTS[
    'David will work for several years collecting information, e.g. about brass bands, and populating a database with this information, as a preparation for writing a book.'
] = (
    "David is interested in understanding the social history of music, e.g. who were the musicians, who was the audience, how did a particular musical environment relate to the wider musical environment. He has a particular interest in understanding the environment around music which would not be classified as 'elite' music. He is also interested in people's experience of listening to music. David will work for several years collecting information, e.g. about brass bands, and populating a database with this information, as a preparation for writing a book. David's work could be helped by:\n\n"
    "sonic visualization, e.g. to recognise 'thumbprints' to identify the composer of a piece of music;\n\n"
    "search of document database using RDF annotations and query-time reasoning;\n\n"
    "NLP, e.g. to suggest relevant keywords for annotation; including using some semantics, e.g. proposing an annotation which does not occur in the text."
)

# ── 4. Amy artistic/technical trends ─────────────────────────────────────────
REPLACEMENTS[
    'During the project, Amy researches both artistic and technical trends surrounding organs in the Netherlands. Amy is interested in finding connections between different regions and time periods to see how the trends have developed. She also wants to put her findings in a historical and social context. Amy hopes that the knowledge graphs can help her identify initial trends. That way she knows where to dig deeper to understand the trends. In order to discover trends, comparisons similar to those needed by the organ advisor (Paul) are required. But besides the comparison of stops in organs made by the same organ builder, the comparisons can be related to almost every component of the organ. For example, the different arthistorical details on the fronts of the organs, or the wind supply system are interesting to compare.'
] = (
    'For her next research project, Amy wants to discover artistic and technical trends of organs and how these developed. The development of these trends could possibly indicate wider social trends. During the project, Amy researches both artistic and technical trends surrounding organs in the Netherlands. Amy is interested in finding connections between different regions and time periods to see how the trends have developed. She also wants to put her findings in a historical and social context. Amy hopes that the knowledge graphs can help her identify initial trends. That way she knows where to dig deeper to understand the trends. In order to discover trends, comparisons similar to those needed by the organ advisor (Paul) are required. But besides the comparison of stops in organs made by the same organ builder, the comparisons can be related to almost every component of the organ. For example, the different arthistorical details on the fronts of the organs, or the wind supply system are interesting to compare.\n'
    '    Amy already has access to the physical as well as online library of the university where she works.\n'
    '    Amy also has the entire 15 volume collection from the Dutch organ encyclopaedia. She uses the encyclopaedia mostly for reading about the arthistorical and technical aspects and developments of organs and comparing these to other organs.\n'
    '    Database websites for information about organs and churches which all offer slightly different kinds of information:\n'
    '        https://www.kerk-en-orgel.nl\n'
    '        https://reliwiki.nl/\n'
    '        http://www.orgbase.nl\n'
    '        https://geoplaza.vu.nl/cms/research/kerkenkaart/ \u2013 Kerkenkaart (churchmap) from the Vrije Universiteit Amsterdam\n'
)

# ── 5. Paul uncertain restoration ────────────────────────────────────────────
REPLACEMENTS[
    'In this scenario Paul has to plan a similar restoration. However, from his already extensive knowledge about organs, he is uncertain about some of the information that was provided by the encyclopaedia. That is, some of the mentioned components and technicalities such as the pitch seemed illogical when compared to the other organs from the same organ builder. If it would somehow be possible to mark facts that are unreliable or untrue, that would save Paul a lot of time checking, verifying, and notating.'
] = (
    'Successfully plan the restoration of an organ.. In this scenario Paul has to plan a similar restoration. However, from his already extensive knowledge about organs, he is uncertain about some of the information that was provided by the encyclopaedia. That is, some of the mentioned components and technicalities such as the pitch seemed illogical when compared to the other organs from the same organ builder. If it would somehow be possible to mark facts that are unreliable or untrue, that would save Paul a lot of time checking, verifying, and notating.\n'
    '    Het Historische Orgel in Nederland \u2013 Dutch organ encyclopedia\n'
    '    http://www.orgbase.nl \u2013 Dutch hobbyist-made website with information similar to Het Historische Orgel\n'
    '    Church and city archives\n'
    '    Old newspapers\n'
    '    https://www.delpher.nl \u2013 digital database from the Dutch Royal Library\n'
    '    Organs (sight visits)\n'
    '    http://www.dtbob.org/ \u2013 Belgian digital organ database\n'
)

# ── 6. Jorge digital library of scores ───────────────────────────────────────
REPLACEMENTS[
    "Jorge manages a digital library of scores. He aims at describing each score with a rich set of contextual information, although a comprehensive descritpion is often not possible. Among these information, one finds\n\n    The musical work (or 'Opus') this score belongs to. A score can cover a whole Opus (e.g., a symphony) or only a part of it (e.g., the third movement). Additionally, it can contain only some parts, all the parts, a trasnscription for non-original instruments, etc. Jorge wants to preserve each score at the appropriate level, with an adequate referencing (e.g., the official Opus number in a standard catalogue exists, K234.a, of BWV192, etc.)\n    Licence and copyright information\n    Relations to standard external resources to refer to, e.g., composers or other kind of authorships\n    An organisation in collections, clear and flexible\n    Tools to manage this organisation and search for relevant scores."
] = (
    "The dashboard presents an overview of the classifying dimensions. Some of them might be hierarchical in nature (for instance the region/country/city classification). Whenever a dimension is chosen, the dashboard adapts to this new context by refining the classification of scores based on sub-dimensions. Jorge manages a digital library of scores. He aims at describing each score with a rich set of contextual information, although a comprehensive descritpion is often not possible. Among these information, one finds\n\n"
    "    The musical work (or 'Opus') this score belongs to. A score can cover a whole Opus (e.g., a symphony) or only a part of it (e.g., the third movement). Additionally, it can contain only some parts, all the parts, a trasnscription for non-original instruments, etc. Jorge wants to preserve each score at the appropriate level, with an adequate referencing (e.g., the official Opus number in a standard catalogue exists, K234.a, of BWV192, etc.)\n"
    "    Licence and copyright information\n"
    "    Relations to standard external resources to refer to, e.g., composers or other kind of authorships\n"
    "    An organisation in collections, clear and flexible\n"
    "    Tools to manage this organisation and search for relevant scores."
    "In terms of digital tools, Jorge wants a dashboard that gives him at a glance on overview of a score library content. The overview shows statistics on the library organized after the many dimensions that can be used to classify score: composer, period, countries/region/city, style, length, tonality, prominent patterns, orchestration, etc."
)

# ── 7. Ortenz prosopographic ──────────────────────────────────────────────────
REPLACEMENTS[
    "Ortenz is supervising a PhD student. They want to explore a database of prosopographic information of personalities relevant to the musical cultural heritage, focusing primarily on musicians' career but also involving relevant people in sectors such as art, politics, and industry. They are interested in the events and facts nad how they are linked to the sources (biographies, letters, memoirs, encyclopedia). She wants the system to allow her to make annotations on the content, rate the quality of the sources and the accuracy of the statements, and curate collections of facts/statements/events as material for scholarship."
] = (
    "Ortenz would like to have a system for visualising events (meetings of composers and musicians) in time and space in order to track musicians' careers, their overlap and intersections, gathering trends in time and space, and making emerge patterns of knowledge transmission."
    "Ortenz is supervising a PhD student. They want to explore a database of prosopographic information of personalities relevant to the musical cultural heritage, focusing primarily on musicians' career but also involving relevant people in sectors such as art, politics, and industry. They are interested in the events and facts nad how they are linked to the sources (biographies, letters, memoirs, encyclopedia). She wants the system to allow her to make annotations on the content, rate the quality of the sources and the accuracy of the statements, and curate collections of facts/statements/events as material for scholarship. \n"
    "    Wikipedia\n"
    "    DBpedia\n"
    "    Wikidata\n"
)

# ── 8. Paul first restoration ─────────────────────────────────────────────────
REPLACEMENTS[
    'Paul is hired to plan and supervise the restoration of an organ in a church. The church is from the sixteenth century. Paul would look up the organ in the Dutch organ encyclopaedia, which is ordered based on the year of production of the case of the organ. Because the organ is not necessarily made in the same year as the church, he first has to go through the indexes of the encyclopaedias in order to find the correct organ. Then, Paul searches for all organs that are made by the same organ builder. These organs are used as material for comparisons of the technical features. Finding and comparing the organs can be a tedious and time-consuming task. If this can be simplified and sped up by the use of the portal, Paul would save a lot of time and effort. As it turns out, the comparison of the disposition indicates that the organ up for restoration has been changed significantly during a previous restoration. This is indicated by a large difference in base stops that is used in all organs by the same organ builder. Paul now has to go visit some of the other organs and the church archives in order to verify whether some of those stops were the original stops or not.'
] = (
    'Successfully plan the restoration of an organ? Paul is hired to plan and supervise the restoration of an organ in a church. The church is from the sixteenth century. Paul would look up the organ in the Dutch organ encyclopaedia, which is ordered based on the year of production of the case of the organ. Because the organ is not necessarily made in the same year as the church, he first has to go through the indexes of the encyclopaedias in order to find the correct organ. Then, Paul searches for all organs that are made by the same organ builder. These organs are used as material for comparisons of the technical features. Finding and comparing the organs can be a tedious and time-consuming task. If this can be simplified and sped up by the use of the portal, Paul would save a lot of time and effort. As it turns out, the comparison of the disposition indicates that the organ up for restoration has been changed significantly during a previous restoration. This is indicated by a large difference in base stops that is used in all organs by the same organ builder. Paul now has to go visit some of the other organs and the church archives in order to verify whether some of those stops were the original stops or not.\n'
    '    Het Historische Orgel in Nederland \u2013 Dutch organ encyclopedia\n'
    '    http://www.orgbase.nl \u2013 Dutch hobbyist-made website with information similar to Het Historische Orgel\n'
    '    Church and city archives\n'
    '    Old newspapers\n'
    '    https://www.delpher.nl \u2013 digital database from the Dutch Royal Library\n'
    '    Organs (sight visits)\n'
    '    http://www.dtbob.org/ \u2013 Belgian digital organ database\n'
)

# ── 9. Sophia social-historical ───────────────────────────────────────────────
REPLACEMENTS[
    'Sophia is doing social-historical research, using textual sources. One of her current projects is concerned with the relationship between music, medicine and religion at a particular charitable institution in 17th century Italy. The sources she uses, which are also not digitized, include records of an institution, payslips etc.'
] = (
    "Sophia is a musicologist and a practising musician. Sophia is interested in understanding the social-historical reasons behind how the music was created and how it sounds. Sophia is doing social-historical research, using textual sources. One of her current projects is concerned with the relationship between music, medicine and religion at a particular charitable institution in 17th century Italy. The sources she uses, which are also not digitized, include records of an institution, payslips etc."
    "The resources which Sophia uses are often not digitized. Therefore she needs to visit libraries. Whilst there, she will sift through the appropriate material, and take photographs of the most relevant. She will then analyze these photographs on her return home. In one particular library she may not be able to take photographs, and there she can only take notes. Sophia uses spreadsheets to maintain and compare information.\n\n"
    "Sophia's work could be helped by:\n\n"
    "    digitalisation of her source material;\n"
    "    handwriting OCR;\n"
    "    data visualization, e.g. to illustrate relationships between composers, institutions etc.;\n"
    "    use of a database, i.e. as upgrade to current use of spreadsheets;\n"
    "    named-entity recognition\n"
)

# ── 10. Mark Dutch folk tunes ─────────────────────────────────────────────────
REPLACEMENTS[
    'Starting from a collection of Dutch folk tunes, Mark attempts to relate these tunes to other documented music, using a variety of databases. Mark would like to relate individual pieces of music, but also repertoires generally. He would also like to understand musical evolution and transmission over time.'
] = (
    "Mark is interested in understanding how Dutch folk tunes relate to other music, e.g. from French court operas."
    "Starting from a collection of Dutch folk tunes, Mark attempts to relate these tunes to other documented music, using a variety of databases. Mark would like to relate individual pieces of music, but also repertoires generally. He would also like to understand musical evolution and transmission over time. Mark works with:\n\n"
    "RISM index of musical sources, composers etc https://opac.rism.info/main-menu-/kachelmenu with the ability to search on names and music\n\n"
    "NEUMA, a digital library of musical scores http://neuma.huma-num.fr/home/presentation\n\n"
    "ABC notation database http://abcnotation.com/\n\n"
    "Database of Dutch tunes http://www.liederenban.nl and http://www.liederebank.nl/mtc\n\n"
    "Early American Secular Music and its European sources http://www.cdss.org/elibrary/Easmes/Index.htm\n\n"
    "Mark's work could be helped by:\n\n"
    "Connection to other databases, so that he could automatically query a range of databases"
)

# ── Apply all replacements ────────────────────────────────────────────────────
replaced = 0
for old, new in REPLACEMENTS.items():
    mask = df['Scenario'] == old
    count = mask.sum()
    if count == 0:
        print(f"WARNING: no match for scenario starting with: {old[:60]!r}")
    else:
        df.loc[mask, 'Scenario'] = new
        print(f"Replaced {count} row(s): {old[:60]!r}")
        replaced += 1

print(f"\nTotal scenarios updated: {replaced}/{len(REPLACEMENTS)}")

df.to_csv(CSV_PATH, index=False, encoding='utf-8')
print(f"Saved to {CSV_PATH}")
