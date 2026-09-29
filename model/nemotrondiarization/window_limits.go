package nemotrondiarization

// The largest complete 9+4 prepared streaming window has 264 speaker-cache
// rows, 264 FIFO rows, nine current rows and four lookahead rows. Attention
// and the speaker head process the whole window without truncating context.
const maxPreparedDiarizationRows = 2*diarizationStreamFIFO + 9 + 4
