#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <Eigen/Dense>
#include <fstream>
#include <sstream>

using namespace std;
using namespace Eigen;

const int vocab_size=37000;
const int dimensions=512;
const int max_seq_len=190;
const int num_heads=8;
const int d_k=dimensions/num_heads;
const int ff_hidden_dim=2048; // same as in papaer 


void read_file(string filename,vector<vector<int>>&source_tokens,vector<vector<int>>&target_tokens)
{
    fstream file(filename);
    string line;
    while (getline(file,line))
    {
        stringstream ss(line);
        string source_str,target_str;
        getline(ss,source_str,'\t');
        getline(ss,target_str,'\t');

        vector<int>source_line;
        stringstream source_stream(source_str);
        int token_id;

        while (source_stream >>token_id)
        {
            source_line.push_back(token_id);
        }
        source_tokens.push_back(source_line);

        vector<int>target_line;
        stringstream target_stream(target_str);
        while (target_stream>>token_id)
        {
            target_line.push_back(token_id);
        }
        target_tokens.push_back(target_line);
    }
    
}


MatrixXf initialize_embedding_matrix(int vocab_size,int dimensions)
{
    MatrixXf embedding(vocab_size,dimensions);
    default_random_engine generator;
    normal_distribution<float>distribution(0.0,0.02);
    for(int i=0;i<vocab_size;i++)
    {
        for(int j=0;j<dimensions;j++)
        {
            embedding(i,j)=distribution(generator);
        }
    }
    return embedding;
}

MatrixXf get_embeddings(const vector<int>&token_ids,const MatrixXf& embedding_matrix)
{
    int seq_len=token_ids.size();
    MatrixXf embedded_vector(seq_len,dimensions);
    for(int i=0;i<seq_len;i++)
    {
        embedded_vector.row(i)=embedding_matrix.row(token_ids[i]);
    }
    return embedded_vector;
}

MatrixXf get_positional_encoding(int max_len,int dimensions)
{
    MatrixXf positional_encoding(max_len,dimensions);
    for(int pos=0;pos<max_len;pos++)
    {
        for(int i=0;i<dimensions;i++)
        {
            float angle=pos/pow(10000.0,(2*(i/2))/float(dimensions));
            positional_encoding(pos,i)=(i%2==0?sin(angle):cos(angle));
        }
    }
    return positional_encoding;
}

MatrixXf add_positional_encoding(const MatrixXf& embeddings,const MatrixXf& positional_encoding)
{
    int seq_len=embeddings.rows();
    return embeddings+positional_encoding.topRows(seq_len);
}


vector<int>padding(const vector<int>&input,int pad_id)
{
    vector<int>padded=input;
    if(padded.size()<max_seq_len)
    {
        padded.resize(max_seq_len,pad_id);
    }
    else if(padded.size()>max_seq_len)
    {
        padded=vector<int>(padded.begin(),padded.begin()+max_seq_len);
    }
    return padded;
}

VectorXf create_padding_mask(const vector<int>&input,int pad_id)
{
    VectorXf mask(max_seq_len);
    for(int i=0;i<max_seq_len;i++)
    {
        mask(i)=(input[i]==pad_id?0.0f:1.0f);
    }
    return mask;
}


MatrixXf initialise_weight_matrixes(int d_model,int d_k)  //intialises W_k, W_q,W_v
{
    MatrixXf W(d_model,d_k);
    default_random_engine generator;
    normal_distribution<float>distribution(0.0,0.02);
    for(int i=0;i<d_model;i++)
    {
        for(int j=0;j<d_k;j++)
        {
            W(i,j)=distribution(generator);
        }
    }
    return W;
}

MatrixXf compute_QKV(const MatrixXf& X,const MatrixXf& W)
{
    return X*W;
}

MatrixXf scaled_dot_product_attention(const MatrixXf& Q, const MatrixXf& K , const MatrixXf& V,const VectorXf& mask)
{
    int sq_len=Q.rows();
    MatrixXf scores=Q*K.transpose();
    scores/=sqrt(d_k);
    for(int i=0;i<sq_len;i++)
    {
        for (int j = 0; j < sq_len; j++)
        {
            if(mask(j)==0.0f)
            {
                scores(i,j)=-1e9;
            }
        }
    }
    for(int i=0;i<sq_len;i++)
    {
        float max_score=scores.row(i).maxCoeff();
        VectorXf exps=(scores.row(i).array()-max_score).exp();
        float sum_exp=exps.sum();
        scores.row(i)=exps/sum_exp;
    }
    return scores*V;
}

MatrixXf multi_head_attention(const MatrixXf& X,const VectorXf& mask,const vector<MatrixXf>& W_q_heads,const vector<MatrixXf>& W_k_heads,const vector<MatrixXf>& W_v_heads,const MatrixXf& W_o)
{
    int seq_len=X.rows();
    vector<MatrixXf>head_outputs;
    for(int i=0;i<num_heads;i++)
    {
        MatrixXf Q=compute_QKV(X,W_q_heads[i]);
        MatrixXf K=compute_QKV(X,W_k_heads[i]);
        MatrixXf V=compute_QKV(X,W_v_heads[i]);
        MatrixXf attention_output = scaled_dot_product_attention(Q, K, V, mask);
        head_outputs.push_back(attention_output);
    }
    MatrixXf concat(seq_len,dimensions);
    for(int i=0;i<num_heads;i++)
    {
        concat.block(0,i*d_k,seq_len,d_k)=head_outputs[i];
    }
    return concat*W_o;
}


MatrixXf residual_layer_norm(const MatrixXf& input,const MatrixXf& sublayer_output)
{
    MatrixXf combined=input+sublayer_output;
    MatrixXf normalised(combined.rows(),combined.cols());
    float epsilon=1e-6;
    for(int i=0;i<combined.rows();i++)
    {
        VectorXf row=combined.row(i);
        float mean=row.mean();
        float variance=(row.array()-mean).square().mean();
        VectorXf normalized=(row.array()-mean)/sqrt(variance+epsilon);
        normalised.row(i)=normalized;
    }   
    return normalised;
}

VectorXf init_ff_bias(int dimension)
{
    return VectorXf::Zero(dimension);
}

MatrixXf feed_forward(const MatrixXf& X,const MatrixXf& W1,const VectorXf& b1,const MatrixXf& W2,const VectorXf& b2)
{
    MatrixXf hidden=(X*W1).rowwise()+b1.transpose();
    for(int i=0;i<hidden.rows();i++)
    {
        for(int j=0;j<hidden.cols();j++)
        {
            hidden(i,j)=max(0.0f,hidden(i,j));
        }
    }
    MatrixXf output=(hidden*W2).rowwise()+b2.transpose();
    return output;
}


struct Encoder_layer_parameters
{
    vector<MatrixXf> W_q_heads,W_k_heads,W_v_heads;
    MatrixXf W_o;
    MatrixXf W1;
    MatrixXf W2;
    VectorXf b1;
    VectorXf b2;
};

Encoder_layer_parameters initialise_encoder_layer()
{
    Encoder_layer_parameters parameters;
    for(int i=0;i<num_heads;i++)
    {
        parameters.W_q_heads.push_back(initialise_weight_matrixes(dimensions,d_k));
        parameters.W_k_heads.push_back(initialise_weight_matrixes(dimensions,d_k));
        parameters.W_v_heads.push_back(initialise_weight_matrixes(dimensions,d_k));
    }
    parameters.W_o=initialise_weight_matrixes(dimensions,dimensions);
    parameters.W1=initialise_weight_matrixes(dimensions,ff_hidden_dim);
    parameters.W2=initialise_weight_matrixes(ff_hidden_dim,dimensions);
    parameters.b1=init_ff_bias(ff_hidden_dim);
    parameters.b2=init_ff_bias(dimensions);
    return parameters;
}

MatrixXf encoder_layer(const MatrixXf& X,const VectorXf &mask,const Encoder_layer_parameters& layer)
{
    MatrixXf attention_output = multi_head_attention(X,mask,layer.W_q_heads,layer.W_k_heads,layer.W_v_heads,layer.W_o);
    MatrixXf normalised_output=residual_layer_norm(X,attention_output);

    MatrixXf ffn_output=feed_forward(normalised_output,layer.W1,layer.b1,layer.W2,layer.b2);
    MatrixXf final_ouput=residual_layer_norm(normalised_output,ffn_output);
    return final_ouput;
}

MatrixXf create_causal_mask(int seq_len)
{
    MatrixXf mask=MatrixXf::Zero(seq_len,seq_len);
    for(int i=0;i<seq_len;i++)
    {
        for(int j=0;j<=i;j++)
        {
            mask(i,j)=1.0f;
        }
    }
    return mask;
}

struct Decoder_layer_parameters
{
    vector<MatrixXf> W_q_heads,W_k_heads,W_v_heads;
    vector<MatrixXf> W_q_cross,W_k_cross,W_v_cross;
    MatrixXf W_o_self;
    MatrixXf W_o_cross;
    MatrixXf W1; MatrixXf W2;
    VectorXf b1; VectorXf b2;
};

Decoder_layer_parameters initialise_decoder_layer()
{
    Decoder_layer_parameters params;
    for (int i = 0; i < num_heads; ++i)
    {
        params.W_q_heads.push_back(initialise_weight_matrixes(dimensions, d_k));
        params.W_k_heads.push_back(initialise_weight_matrixes(dimensions, d_k));
        params.W_v_heads.push_back(initialise_weight_matrixes(dimensions, d_k));

        params.W_q_cross.push_back(initialise_weight_matrixes(dimensions, d_k));
        params.W_k_cross.push_back(initialise_weight_matrixes(dimensions, d_k));
        params.W_v_cross.push_back(initialise_weight_matrixes(dimensions, d_k));
    }

    params.W_o_self = initialise_weight_matrixes(dimensions, dimensions);
    params.W_o_cross = initialise_weight_matrixes(dimensions, dimensions);
    params.W1 = initialise_weight_matrixes(dimensions, ff_hidden_dim);
    params.W2 = initialise_weight_matrixes(ff_hidden_dim, dimensions);
    params.b1 = init_ff_bias(ff_hidden_dim);
    params.b2 = init_ff_bias(dimensions);
    return params;
}

MatrixXf decoder_layer(const MatrixXf& target,const MatrixXf& encoder_output,const VectorXf& pad_mask,const MatrixXf& causal_mask,const Decoder_layer_parameters& layer)
{
    MatrixXf padding_mask=pad_mask*pad_mask.transpose();
    MatrixXf attn_mask=causal_mask.cwiseProduct(padding_mask);
    
    MatrixXf self_attn=multi_head_attention(target,attn_mask,layer.W_q_heads,layer.W_k_heads,layer.W_v_heads,layer.W_o_self);
    MatrixXf norm1=residual_layer_norm(target,self_attn);

    MatrixXf encoder_attn=multi_head_attention(norm1,pad_mask,layer.W_q_cross,layer.W_k_cross,layer.W_v_cross,layer.W_o_cross);
    MatrixXf norm2=residual_layer_norm(norm1,encoder_attn);

    MatrixXf ffn_output=feed_forward(norm2,layer.W1,layer.b1,layer.W2,layer.b2);
    MatrixXf final_output=residual_layer_norm(norm2,ffn_output);
    return final_output;
}

vector<int> shift_right(const vector<int>& tgt) {
    vector<int> shifted = {1}; // <sos>
    for (int i = 0; i < tgt.size() - 1; ++i)
        shifted.push_back(tgt[i]);
    return padding(shifted, 0); // pad it
}

int main()
{
    vector<vector<int>>source_tokens;
    vector<vector<int>>target_tokens;
    vector<VectorXf>padding_masks;
    read_file("tokenized_en_de.txt",source_tokens,target_tokens);
    MatrixXf embedding_matrix=initialize_embedding_matrix(vocab_size,dimensions);
    MatrixXf positional_encoding=get_positional_encoding(max_seq_len,dimensions);
   
    const int num_layers=6;
    vector<Encoder_layer_parameters>encoder_stack;
    vector<Decoder_layer_parameters>decoder_stack;
    for(int i=0;i<num_layers;i++)
    {
        encoder_stack.push_back(initialise_encoder_layer());
        decoder_stack.push_back(initialise_decoder_layer());
    }
    for(int i=0;i<source_tokens.size();i++)
    {
        vector<int>padded_token=padding(source_tokens[i],0);
        MatrixXf X=get_embeddings(padded_token,embedding_matrix);
        X=add_positional_encoding(X,positional_encoding);
        VectorXf mask=create_padding_mask(padded_token,0);
        MatrixXf encoder_output=X;
        for(int l=0;l<num_layers;l++)
        {

            X=encoder_layer(X,mask,encoder_stack[l]);
        }
        MatrixXf encoder_output=X;
        vector<int>targer_padded=shift_right(target_tokens[i]);
        MatrixXf decoder_input=get_embeddings(targer_padded,embedding_matrix);
        decoder_input=add_positional_encoding(decoder_input,positional_encoding);
        VectorXf target_padded_mask=create_padding_mask(targer_padded,0);
        MatrixXf causal_mask=create_causal_mask(max_seq_len);
        MatrixXf decoder_output=decoder_input;
        for(int l=0;l<num_layers;l++)
        {
            decoder_output=decoder_layer(decoder_output,encoder_output,target_padded_mask,causal_mask,decoder_stack[l]);
        }
        MatrixXf W_vocab = initialise_weight_matrixes(dimensions, vocab_size);
        MatrixXf logits = decoder_output * W_vocab;
    }

    return 0;
}