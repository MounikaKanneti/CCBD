import java.io.IOException;
import java.util.HashMap;
import java.util.Scanner;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.*;
import org.apache.hadoop.mapreduce.lib.output.*;

class Element implements Writable {
    private int tag;
    private int index;
    private double value;

    public Element() {
        this.tag = 0;
        this.index = 0;
        this.value = 0.0;
    }

    public Element(int tag, int index, double value) {
        this.tag = tag;
        this.index = index;
        this.value = value;
    }

    @Override
    public void readFields(DataInput input) throws IOException {
        this.tag = input.readInt();
        this.index = input.readInt();
        this.value = input.readDouble();
    }

    @Override
    public void write(DataOutput output) throws IOException {
        output.writeInt(this.tag);
        output.writeInt(this.index);
        output.writeDouble(this.value);
    }
}

class Pair implements WritableComparable<Pair> {
    public int i;
    public int j;

    public Pair() {
        i = 0;
        j = 0;
    }

    public Pair(int i, int j) {
        this.i = i;
        this.j = j;
    }

    @Override
    public void readFields(DataInput input) throws IOException {
        this.i = input.readInt();
        this.j = input.readInt();
    }

    @Override
    public void write(DataOutput output) throws IOException {
        output.writeInt(this.i);
        output.writeInt(this.j);
    }
}

public class Multiply extends Configured implements Tool {

    @Override
    public int run(String[] args) throws Exception {
        return 0;
    }

    // Mapper class for M-matrix
    public static class Mapper_M extends Mapper<Object, Text, Pair, Text> {
        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            Scanner scanner = new Scanner(value.toString()).useDelimiter(",");
            int a = scanner.nextInt();
            int b = scanner.nextInt();
            double z = scanner.nextDouble();
            String str = "M" + Double.toString(z);
            context.write(new Pair(a, b), new Text(str));
            scanner.close();
        }
    }

    // Mapper class for N-matrix
    public static class Mapper_N extends Mapper<Object, Text, Pair, Text> {
        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            Scanner scanner = new Scanner(value.toString()).useDelimiter(",");
            int a = scanner.nextInt();
            int b = scanner.nextInt();
            double z = scanner.nextDouble();
            String s = "N" + Double.toString(z);
            context.write(new Pair(a, b), new Text(s));
            scanner.close();
        }
    }

    // Reducer class for M&N matrix
    public static class Reducer_MN extends Reducer<Pair, Text, Text, DoubleWritable> {
        @Override
        public void reduce(Pair key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            HashMap<Pair, Double> G = new HashMap<>();
            HashMap<Pair, Double> H = new HashMap<>();
            
            for (Text t : values) {
                String val = t.toString();
                Double z = Double.parseDouble(val.substring(1));
                
                if (val.startsWith("M")) {
                    G.put(new Pair(key.i, key.j), z);
                } else {
                    H.put(new Pair(key.i, key.j), z);
                }
            }

            for (Pair m : G.keySet()) {
                for (Pair n : H.keySet()) {
                    if (m.j == n.i) {
                        Double q = G.get(m) * H.get(n);
                        String valued = String.valueOf(m.i) + " " + String.valueOf(n.j);
                        context.write(new Text(valued), new DoubleWritable(q));
                    }
                }
            }
        }
    }

    // Mapper class for Mapper_2
    public static class Mapper_2 extends Mapper<Object, Text, Text, DoubleWritable> {
        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            Scanner scanner = new Scanner(value.toString()).useDelimiter("\\s+");
            int a = scanner.nextInt();
            int b = scanner.nextInt();
            double y = scanner.nextDouble();
            context.write(new Text(String.valueOf(a) + "," + String.valueOf(b)), new DoubleWritable(y));
            scanner.close();
        }
    }

    // Reducer class for Reducer_2
    public static class Reducer_2 extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {
        @Override
        public void reduce(Text key, Iterable<DoubleWritable> values, Context context) throws IOException, InterruptedException {
            double sumValue = 0.0;
            for (DoubleWritable value : values) {
                sumValue += value.get();
            }
            context.write(key, new DoubleWritable(sumValue));
        }
    }

    // Job class helps to configure and execute the class
    public static void main(String[] args) throws Exception {
        // Create a new Job
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "MapIntermediate");
        
        // Set job-specific parameters
        job.setJarByClass(Multiply.class);
        job.setMapperClass(Mapper_M.class);
        job.setReducerClass(Reducer_MN.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);
        job.setMapOutputKeyClass(Pair.class);
        job.setMapOutputValueClass(Text.class);
        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        // Set input and output paths
        MultipleInputs.addInputPath(job, new Path(args[0]), TextInputFormat.class, Mapper_M.class);
        MultipleInputs.addInputPath(job, new Path(args[1]), TextInputFormat.class, Mapper_N.class);
        FileOutputFormat.setOutputPath(job, new Path(args[2]));

        // Submit the job and wait for completion
        boolean success = job.waitForCompletion(true);

        if (success) {
            // Create a new Job
            Job job2 = Job.getInstance(conf, "MapFinalOutput");
            
            // Set job-specific parameters
            job2.setJarByClass(Multiply.class);
            job2.setMapperClass(Mapper_2.class);
            job2.setReducerClass(Reducer_2.class);
            job2.setOutputKeyClass(Text.class);
            job2.setOutputValueClass(DoubleWritable.class);
            job2.setMapOutputKeyClass(Text.class);
            job2.setMapOutputValueClass(DoubleWritable.class);
            job2.setInputFormatClass(TextInputFormat.class);
            job2.setOutputFormatClass(TextOutputFormat.class);
            
            // Set input and output paths
            FileInputFormat.setInputPaths(job2, new Path(args[2]));
            FileOutputFormat.setOutputPath(job2, new Path(args[3]));
            
            // Submit the job and wait for completion
            job2.waitForCompletion(true);
        }
    }
}