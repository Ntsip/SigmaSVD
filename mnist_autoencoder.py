class AutoEncoder(Module):
    '''A class that implements an AutoEncoder
    '''
    @staticmethod
    def get_non_linearity(params):
        '''Determine which non linearity is to be used for both encoder and decoder'''
        def get_one(param):
            '''Determine which non linearity is to be used for either encoder or decoder'''
            param = param.lower()
            if param=='relu': return ReLU()
            if param=='sigmoid': return Sigmoid()
            return None

        decoder_non_linearity = get_one(params[0])
        # encoder_non_linearity = getnl(params[a]) if len(params)>1 else decoder_non_linearity
        encoder_non_linearity = get_one((params[a]) for a in params) if len(params)>1 else decoder_non_linearity

        return encoder_non_linearity,decoder_non_linearity

    @staticmethod
    def build_layer(sizes,
                    non_linearity = None):
        '''Construct encoder or decoder as a Sequential of Linear labels, with or without non-linearities

        Positional arguments:
            sizes   List of sizes for each Linear Layer
        Keyword arguments:
            non_linearity  Object used to introduce non-linearity between layers
        '''
        linears = [Linear(m,n) for m,n in zip(sizes[:-1],sizes[1:])]
        if non_linearity==None:
            return Sequential(*linears)
        else:
            return Sequential(*[item for pair in [(layer,non_linearity) for layer in linears] for item in pair])

    def __init__(self,
                 encoder_sizes         = [28*28,400,200,100,50,25,6],
                 encoder_non_linearity = ReLU(inplace=True),
                 decoder_sizes         = [],
                 decoder_non_linearity = ReLU(inplace=True)):
        '''
        Keyword arguments:
            encoder_sizes            List of sizes for each Linear Layer in encoder
            encoder_non_linearity    Object used to introduce non-linearity between encoder layers
            decoder_sizes            List of sizes for each Linear Layer in decoder
            decoder_non_linearity    Object used to introduce non-linearity between decoder layers
        '''
        super().__init__()
        self.encoder_sizes = encoder_sizes
        self.decoder_sizes = encoder_sizes[::-1] if len(decoder_sizes)==0 else decoder_sizes


        self.encoder = AutoEncoder.build_layer(self.encoder_sizes,
                                               non_linearity = encoder_non_linearity)
        self.decoder = AutoEncoder.build_layer(self.decoder_sizes,
                                               non_linearity = decoder_non_linearity)
        self.encode  = True
        self.decode  = True


    def forward(self, x):
        '''Propagate value through network

           Computation is controlled by self.encode and self.decode
        '''
        if self.encode:
            x = self.encoder(x)

        if self.decode:
            x = self.decoder(x)
        return x

    def n_encoded(self):
        return self.encoder_sizes[-1]
